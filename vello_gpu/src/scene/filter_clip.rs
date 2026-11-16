// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Final XY-filter clips are recorded in parent coordinates, before the source shift.

use super::Scene;
use vello_common::geometry::RectU16;
use vello_common::kurbo::{Affine, BezPath, Rect, Shape};
use vello_common::record::LayerClip;
use vello_common::util::strip_bbox;

impl Scene {
    pub(super) fn filter_output_clip(
        &mut self,
        path: Option<&BezPath>,
        transform: Affine,
    ) -> LayerClip {
        // At this point both generator and active clips still belong to the parent.
        let viewport = Rect::new(
            0.0,
            0.0,
            f64::from(self.viewport_state.width()),
            f64::from(self.viewport_state.height()),
        )
        .to_path(0.1);
        let (path, transform) =
            path.map_or((&viewport, Affine::IDENTITY), |path| (path, transform));
        let mut storage = self.strip_storage.borrow_mut();
        let start = storage.strips.len();
        self.viewport_state
            .with_generator_and_clip(|generator, clip| {
                generator.generate_filled_path(
                    path,
                    self.render_state.fill_rule,
                    transform,
                    self.aliasing_threshold,
                    &mut storage,
                    clip.map(|clip| clip.path),
                );
            });
        let range = start..storage.strips.len();
        LayerClip {
            bbox: strip_bbox(&storage.strips[range.clone()]).unwrap_or(RectU16::ZERO),
            strip_range: range,
            thread_idx: 0,
        }
    }
}

#[cfg(all(test, feature = "ahcut"))]
mod tests {
    use crate::scene::Scene;
    use vello_common::filter_effects::{Filter, FilterPrimitive};
    use vello_common::kurbo::{Affine, Rect, Shape};
    #[test]
    fn nested_filter_output_clip_uses_parent_coordinates_and_preserves_source_halo() {
        let mut scene = Scene::new(160, 120);
        scene.push_clip_rect(&Rect::new(20.0, 12.0, 100.0, 90.0));
        let filter = || {
            Filter::from_primitive(FilterPrimitive::GaussianBlurXY {
                std_deviation: [6.0, 6.0],
                linear_rgb: true,
            })
        };
        scene.set_transform(Affine::translate((28.0, 20.0)));
        let region = Rect::new(0.0, 0.0, 12.0, 16.0).to_path(0.1);
        scene.push_layer(Some(&region), None, None, None, Some(filter()));
        let clip = scene
            .recorder
            .layers
            .last()
            .unwrap()
            .props
            .clip_path
            .as_ref()
            .unwrap();
        assert!(clip.bbox.x0 <= 28 && clip.bbox.x1 >= 40);
        assert!(clip.bbox.y0 <= 20 && clip.bbox.y1 >= 36);
        assert!(
            scene.viewport_state.clip().is_none(),
            "inherited ancestor clips must not trim filter input"
        );
        scene.push_clip_rect(&Rect::new(-8.0, -8.0, 40.0, 40.0));
        let outer_clip = scene.viewport_state.clip().unwrap().path.bbox;
        let expected = scene
            .effective_path_transform()
            .transform_rect_bbox(Rect::new(0.0, 0.0, 12.0, 16.0));
        scene.push_layer(Some(&region), None, None, None, Some(filter()));
        let nested = scene
            .recorder
            .layers
            .last()
            .unwrap()
            .props
            .clip_path
            .as_ref()
            .unwrap()
            .bbox;
        assert!(
            (f64::from(nested.x0) - expected.x0).abs() <= 4.0
                && (f64::from(nested.x1) - expected.x1).abs() <= 4.0
        );
        assert!(
            (f64::from(nested.y0) - expected.y0).abs() <= 4.0
                && (f64::from(nested.y1) - expected.y1).abs() <= 4.0
        );
        assert!(scene.viewport_state.clip().is_none());
        scene.pop_layer();
        assert_eq!(scene.viewport_state.clip().unwrap().path.bbox, outer_clip);
        scene.pop_clip();
        scene.pop_layer();
        assert!(scene.viewport_state.clip().is_some());
        scene.pop_clip();
        assert!(scene.viewport_state.clip().is_none());
    }
}
