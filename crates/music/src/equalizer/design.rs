//! Turns a filter into coefficients at one rate. The writer side reads the RBJ cookbook biquad a
//! filter means, whose magnitude has a closed form. Everything goes through `place`, so the
//! rules for a filter near the Nyquist frequency are decided once.

use std::f64::consts::{LN_2, PI, TAU};

use super::{Filter, Kind, Width};

/// Above this fraction of the rate a filter is designed at its limit or left out, since its
/// effect lies where the stream cannot carry it.
const NYQUIST: f64 = 0.49;

/// What a filter sounds like, without whether it is on. The audio thread interpolates these
/// during a glide and redesigns from them every sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shape {
    pub kind: Kind,
    pub frequency: f64,
    pub gain: f64,
    pub width: Width,
}

impl Shape {
    pub fn of(filter: &Filter) -> Self {
        Self {
            kind: filter.kind,
            frequency: filter.frequency,
            gain: filter.gain,
            width: filter.width,
        }
    }
}

impl Width {
    /// The width as the Q the design uses at `frequency` and `rate`. Octaves convert with the
    /// prewarped relation Equalizer APO uses, which is why they depend on the rate.
    pub(crate) fn q(self, frequency: f64, rate: f64) -> f64 {
        match self {
            Self::Q(q) => q,
            Self::Octaves(octaves) => {
                let w0 = TAU * frequency / rate;
                1. / (2. * (LN_2 / 2. * octaves * w0 / w0.sin()).sinh())
            }
        }
    }
}

/// Where a filter lands at one rate. A filter whose frequency the rate cannot carry is left out,
/// kept as a low shelf's flat gain, or held at the limit for a high pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Placed {
    Identity,
    Flat(f64),
    Filter {
        kind: Kind,
        frequency: f64,
        gain: f64,
        q: f64,
    },
}

pub(crate) fn place(shape: &Shape, rate: f64) -> Placed {
    let limit = NYQUIST * rate;
    let mut frequency = shape.frequency;
    if frequency >= limit {
        match shape.kind {
            Kind::HighPass => frequency = limit,
            Kind::LowShelf => return Placed::Flat(shape.gain),
            _ => return Placed::Identity,
        }
    }
    Placed::Filter {
        kind: shape.kind,
        frequency,
        gain: shape.gain,
        q: shape.width.q(frequency, rate),
    }
}

/// `φ = sin²(πf/fs)`, the one number the closed form for a biquad's magnitude at `frequency`
/// needs.
pub(crate) fn phi(frequency: f64, rate: f64) -> f64 {
    (PI * frequency / rate).sin().powi(2)
}

/// The RBJ cookbook biquad a filter means, normalised so `a0` is one. This is the response the
/// state variable filter reproduces and the one the writer side reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    pub fn of(shape: &Shape, rate: f64) -> Self {
        let flat = |b0: f64| Self {
            b0,
            b1: 0.,
            b2: 0.,
            a1: 0.,
            a2: 0.,
        };
        let (kind, frequency, gain, q) = match place(shape, rate) {
            Placed::Identity => return flat(1.),
            Placed::Flat(db) => return flat(10f64.powf(db / 20.)),
            Placed::Filter {
                kind,
                frequency,
                gain,
                q,
            } => (kind, frequency, gain, q),
        };
        let a = 10f64.powf(gain / 40.);
        let (sn, cs) = (TAU * frequency / rate).sin_cos();
        let alpha = sn / (2. * q);
        let beta = 2. * a.sqrt() * alpha;
        let (b, a) = match kind {
            Kind::LowPass => (
                [(1. - cs) / 2., 1. - cs, (1. - cs) / 2.],
                [1. + alpha, -2. * cs, 1. - alpha],
            ),
            Kind::HighPass => (
                [(1. + cs) / 2., -(1. + cs), (1. + cs) / 2.],
                [1. + alpha, -2. * cs, 1. - alpha],
            ),
            Kind::BandPass => ([alpha, 0., -alpha], [1. + alpha, -2. * cs, 1. - alpha]),
            Kind::Notch => ([1., -2. * cs, 1.], [1. + alpha, -2. * cs, 1. - alpha]),
            Kind::AllPass => (
                [1. - alpha, -2. * cs, 1. + alpha],
                [1. + alpha, -2. * cs, 1. - alpha],
            ),
            Kind::Peak => (
                [1. + alpha * a, -2. * cs, 1. - alpha * a],
                [1. + alpha / a, -2. * cs, 1. - alpha / a],
            ),
            Kind::LowShelf => (
                [
                    a * ((a + 1.) - (a - 1.) * cs + beta),
                    2. * a * ((a - 1.) - (a + 1.) * cs),
                    a * ((a + 1.) - (a - 1.) * cs - beta),
                ],
                [
                    (a + 1.) + (a - 1.) * cs + beta,
                    -2. * ((a - 1.) + (a + 1.) * cs),
                    (a + 1.) + (a - 1.) * cs - beta,
                ],
            ),
            Kind::HighShelf => (
                [
                    a * ((a + 1.) + (a - 1.) * cs + beta),
                    -2. * a * ((a - 1.) + (a + 1.) * cs),
                    a * ((a + 1.) + (a - 1.) * cs - beta),
                ],
                [
                    (a + 1.) - (a - 1.) * cs + beta,
                    2. * ((a - 1.) - (a + 1.) * cs),
                    (a + 1.) - (a - 1.) * cs - beta,
                ],
            ),
        };
        Self {
            b0: b[0] / a[0],
            b1: b[1] / a[0],
            b2: b[2] / a[0],
            a1: a[1] / a[0],
            a2: a[2] / a[0],
        }
    }

    /// The magnitude in decibels at the frequency `phi` stands for, from the closed form in
    /// `φ = sin²(πf/fs)`. The caller works `phi` out once for every biquad of a curve.
    pub fn db(&self, phi: f64) -> f64 {
        let (b0, b1, b2) = (self.b0, self.b1, self.b2);
        let (a1, a2) = (self.a1, self.a2);
        let num = (b0 + b1 + b2).powi(2) - 4. * (b0 * b1 + 4. * b0 * b2 + b1 * b2) * phi
            + 16. * b0 * b2 * phi * phi;
        let den =
            (1. + a1 + a2).powi(2) - 4. * (a1 + 4. * a2 + a1 * a2) * phi + 16. * a2 * phi * phi;
        10. * num.max(0.).log10() - 10. * den.log10()
    }
}
