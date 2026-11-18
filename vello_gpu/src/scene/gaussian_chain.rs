// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Public layer pushes and pops. A serial Gaussian chain (`Filter::gaussian_chain`) is lowered to separately cropped
//! physical layers that one `pop_layer` closes; every other layer is one physical layer, as before.
use super::*;

impl Scene {
    /// Push a new layer with the given properties.
    ///
    /// # Panics
    ///
    /// Panics if `mask` is provided because mask layers are not yet supported.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        // A chain is lowered from its final output toward the source graphic: popping runs stage 0, crops its
        // output into stage 1, then runs and crops stage 1. The outer layer applies the final crop, the caller's
        // clip, blend and opacity once. A graph that is not a valid chain takes the ordinary path.
        let chain = filter.as_ref().and_then(|f| {
            let stages = f.graph.checked_gaussian_chain().ok().flatten()?;
            Some((stages, f.graph.gaussian_final_region()?))
        });
        let Some((stages, final_region)) = chain else {
            self.push_single_layer(clip_path, blend_mode, opacity, mask, filter);
            self.logical_layer_counts.push(1);
            return;
        };
        let final_path = final_region.to_path(0.1);
        self.push_single_layer(Some(&final_path), blend_mode, opacity, mask, None);
        let extra_clip = clip_path.filter(|path| path.elements() != final_path.elements());
        if let Some(path) = extra_clip {
            self.push_single_layer(Some(path), None, None, None, None);
        }
        for (primitive, region) in stages.iter().rev() {
            let crop = region.to_path(0.1);
            self.push_single_layer(
                Some(&crop),
                None,
                None,
                None,
                Some(Filter::from_primitive(primitive.clone())),
            );
        }
        self.logical_layer_counts
            .push((stages.len() + 1 + usize::from(extra_clip.is_some())) as u8);
    }

    /// Pop the last pushed layer: every physical layer that one `push_layer` opened. A pop without a matching
    /// push behaves as a single physical pop.
    pub fn pop_layer(&mut self) {
        let count = self.logical_layer_counts.pop().unwrap_or(1);
        for _ in 0..count {
            self.pop_single_layer();
        }
    }
}
#[cfg(all(test, feature = "ahcut"))]
mod tests {
    use super::*;
    use vello_common::filter_effects::FilterPrimitive;
    #[test]
    fn serial_layers_balance_nested_ancestor_coordinates_and_reset() {
        let mut scene = Scene::new(257, 97);
        scene.set_transform(Affine::translate((8.0, 4.0)));
        let crop = Rect::new(80.0, 8.0, 168.0, 88.0);
        let narrow = Rect::new(120.0, 44.0, 124.0, 52.0);
        let f = Filter::gaussian_chain(
            &[
                (
                    FilterPrimitive::GaussianBlurXY {
                        std_deviation: [1.0, 0.0],
                        linear_rgb: false,
                    },
                    crop,
                ),
                (
                    FilterPrimitive::GaussianBlurXY {
                        std_deviation: [0.0, 0.0],
                        linear_rgb: false,
                    },
                    narrow,
                ),
            ],
            crop,
        )
        .unwrap();
        scene.push_clip_layer(&crop.to_path(0.1));
        let before = scene.effective_path_transform();
        scene.push_filter_layer(f);
        assert_eq!(scene.logical_layer_counts.as_slice(), &[1, 3]);
        scene.pop_layer();
        assert_eq!(scene.effective_path_transform(), before);
        scene.pop_layer();
        assert!(scene.logical_layer_counts.is_empty());
        scene.reset();
        assert!(scene.logical_layer_counts.is_empty());
    }
    fn lifecycle_filter() -> Filter {
        let final_crop = Rect::new(8.0, 8.0, 96.0, 56.0);
        Filter::gaussian_chain(
            &[
                (
                    FilterPrimitive::GaussianBlurXY {
                        std_deviation: [1.0, 0.0],
                        linear_rgb: false,
                    },
                    final_crop,
                ),
                (
                    FilterPrimitive::GaussianBlurXY {
                        std_deviation: [0.0, 1.0],
                        linear_rgb: false,
                    },
                    Rect::new(24.0, 16.0, 28.0, 24.0),
                ),
            ],
            final_crop,
        )
        .unwrap()
    }

    fn push_active_ancestor_and_chain(scene: &mut Scene) {
        scene.push_clip_layer(&Rect::new(8.0, 8.0, 96.0, 56.0).to_path(0.1));
        scene.push_filter_layer(lifecycle_filter());
        assert_eq!(scene.logical_layer_counts.as_slice(), &[1, 3]);
        assert!(scene.recorder.has_layers());
    }

    fn assert_fresh_balanced_push_pop(scene: &mut Scene) {
        let original_transform = scene.effective_path_transform();
        push_active_ancestor_and_chain(scene);
        scene.pop_layer();
        assert_eq!(scene.logical_layer_counts.as_slice(), &[1]);
        assert!(
            scene.recorder.has_layers(),
            "Ancestor must remain after the chain pop"
        );
        assert_eq!(scene.effective_path_transform(), original_transform);
        scene.pop_layer();
        assert!(scene.logical_layer_counts.is_empty());
        assert!(!scene.recorder.has_layers());
        assert_eq!(scene.effective_path_transform(), original_transform);
    }

    fn assert_full_reset_state(scene: &Scene, width: u16, height: u16) {
        assert!(
            scene.logical_layer_counts.is_empty(),
            "Full reset left stale logical layers"
        );
        assert!(
            !scene.recorder.has_layers(),
            "Full reset left real recorder layers"
        );
        assert!(scene.filter.is_none());
        assert_eq!(scene.effective_path_transform(), Affine::IDENTITY);
        assert_eq!((scene.width(), scene.height()), (width, height));
        assert_eq!(
            (scene.viewport_state.width(), scene.viewport_state.height()),
            (width, height)
        );
    }

    #[test]
    fn filter_effect_reset_preserves_active_ancestor_clip() {
        let mut scene = Scene::new(257, 97);
        scene.set_filter_effect(lifecycle_filter());
        scene.push_clip_layer(&Rect::new(8.0, 8.0, 96.0, 56.0).to_path(0.1));
        let transform = scene.effective_path_transform();
        scene.reset_filter_effect();
        assert!(scene.filter.is_none());
        assert_eq!(scene.logical_layer_counts.as_slice(), &[1]);
        assert!(scene.recorder.has_layers());
        assert_eq!(scene.effective_path_transform(), transform);
        scene.pop_layer();
        assert!(scene.logical_layer_counts.is_empty());
        assert!(!scene.recorder.has_layers());
    }

    #[test]
    fn filter_effect_reset_preserves_nested_ancestor_and_chain() {
        let mut scene = Scene::new(257, 97);
        scene.set_transform(Affine::translate((3.0, 5.0)));
        let ancestor_transform = scene.effective_path_transform();
        scene.set_filter_effect(lifecycle_filter());
        push_active_ancestor_and_chain(&mut scene);
        let active_transform = scene.effective_path_transform();
        let active_viewport = (scene.viewport_state.width(), scene.viewport_state.height());
        scene.reset_filter_effect();
        assert!(scene.filter.is_none());
        assert_eq!(scene.logical_layer_counts.as_slice(), &[1, 3]);
        assert!(scene.recorder.has_layers());
        assert_eq!(scene.effective_path_transform(), active_transform);
        assert_eq!(
            (scene.viewport_state.width(), scene.viewport_state.height()),
            active_viewport
        );
        scene.pop_layer();
        assert_eq!(scene.logical_layer_counts.as_slice(), &[1]);
        assert!(scene.recorder.has_layers());
        assert_eq!(scene.effective_path_transform(), ancestor_transform);
        scene.pop_layer();
        assert!(scene.logical_layer_counts.is_empty());
        assert!(!scene.recorder.has_layers());
    }

    #[test]
    fn full_reset_discards_active_layers_then_accepts_fresh_balanced_layers() {
        let mut scene = Scene::new(257, 97);
        scene.set_transform(Affine::translate((3.0, 5.0)));
        scene.set_filter_effect(lifecycle_filter());
        push_active_ancestor_and_chain(&mut scene);
        scene.reset();
        assert_full_reset_state(&scene, 257, 97);
        assert_fresh_balanced_push_pop(&mut scene);
    }

    #[test]
    fn resize_reset_discards_active_layers_then_accepts_fresh_balanced_layers() {
        let mut scene = Scene::new(257, 97);
        scene.set_transform(Affine::translate((3.0, 5.0)));
        scene.set_filter_effect(lifecycle_filter());
        push_active_ancestor_and_chain(&mut scene);
        scene.reset_and_resize(128, 64);
        assert_full_reset_state(&scene, 128, 64);
        assert_fresh_balanced_push_pop(&mut scene);
    }
}
