use crate::theme::ActiveTheme as _;
use gpui::prelude::*;
use gpui::{App, Div, IntoElement, RenderOnce, SharedString, StyleRefinement, Window, div};

/// A small mark beside a title: `E` for explicit content, or a short label such as the one for
/// AI-generated music. It is square for one letter and widens for more.
#[derive(IntoElement)]
pub struct Badge {
    base: Div,
    label: SharedString,
}

impl Badge {
    /// A mark reading `label`, which the caller has already resolved from Fluent.
    pub fn new(label: impl Into<SharedString>) -> Self {
        Self {
            base: div(),
            label: label.into(),
        }
    }

    /// The explicit-content mark.
    pub fn explicit() -> Self {
        Self::new("E")
    }
}

impl Styled for Badge {
    fn style(&mut self) -> &mut StyleRefinement {
        self.base.style()
    }
}

impl RenderOnce for Badge {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let Self { mut base, label } = self;
        let theme = *cx.theme();
        let overrides = std::mem::take(base.style());

        let mut badge = base
            .h_4()
            .min_w_4()
            .px_1()
            .flex()
            .items_center()
            .justify_center()
            .text_xs()
            .text_color(theme.muted_foreground)
            .bg(theme.muted)
            .rounded_xs()
            .child(label);
        badge.style().refine(&overrides);
        badge
    }
}
