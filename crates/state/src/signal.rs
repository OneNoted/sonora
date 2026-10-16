//! What the equalizer screen shows of the sound on its way out: the spectrum of the music ahead
//! of the equalizer. Nothing is analysed unless a screen holds a watch.

use std::sync::Arc;
use std::time::Duration;

use gpui::prelude::*;
use gpui::{Context, Entity, Task};
use music::Level;

use crate::{Playback, PlaybackState};

/// How often the spectrum is read while watched. The analysis moves on every eighth of its
/// window, about 43 ms at 48 kHz, so this is as often as anything new can be drawn.
const SPECTRUM_TICK: Duration = Duration::from_millis(33);

/// The spectrum heard ahead of the equalizer, read while a screen watches it.
pub struct Signal {
    playback: Entity<Playback>,
    /// The latest levels, `None` while nothing is loaded or before a window has filled.
    levels: Option<Vec<Level>>,
}

impl Signal {
    pub fn new(playback: Entity<Playback>) -> Self {
        Self {
            playback,
            levels: None,
        }
    }

    /// The spectrum ahead of the equalizer, in dBFS, while a track is loaded and watched.
    pub fn levels(&self) -> Option<&[Level]> {
        self.levels.as_deref()
    }

    /// Analyses the music ahead of the equalizer for as long as the returned task is held. The
    /// analysis starts afresh, and stops with the task.
    pub fn watch(&mut self, cx: &mut Context<Self>) -> Task<()> {
        let watch = Arc::new(self.playback.read(cx).equalizer_heard().watch());
        self.levels = None;
        cx.notify();
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(SPECTRUM_TICK).await;
                let reading = watch.clone();
                let levels = cx.background_spawn(async move { reading.read() }).await;
                if this.update(cx, |this, cx| this.show(levels, cx)).is_err() {
                    break;
                }
            }
        })
    }

    /// Takes in a read, cleared while nothing is loaded.
    fn show(&mut self, levels: Option<Vec<Level>>, cx: &mut Context<Self>) {
        let loaded = matches!(
            self.playback.read(cx).state(),
            PlaybackState::Playing | PlaybackState::Paused | PlaybackState::Loading
        );
        let levels = levels.filter(|_| loaded);
        if levels != self.levels {
            self.levels = levels;
            cx.notify();
        }
    }
}
