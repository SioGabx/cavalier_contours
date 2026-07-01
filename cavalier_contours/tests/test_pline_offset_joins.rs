//! Tests for the configurable join types (`LineJoin::Miter`, `Round`, `Bevel`) in the parallel
//! offset pipeline. Tests target `create_raw_offset_polyline` directly to isolate join geometry
//! from the slice clipping/stitching stage, plus full `parallel_offset` comparisons for the
//! convex corner case where all join modes coincide.

use cavalier_contours::polyline::{
    LineJoin, PlineCreation, PlineOffsetOptions, PlineSource, PlineSourceMut, Polyline,
    internal::pline_offset::create_raw_offset_polyline,
};

/// Tolerance for vertex position comparisons in raw offset assertions.
const POS_EPS: f64 = 1e-5;

fn raw_offset(pline: &Polyline, offset: f64, join_type: LineJoin, miter_limit: f64) -> Polyline {
    create_raw_offset_polyline(pline, offset, POS_EPS, join_type, miter_limit)
}

/// Assert a vertex position matches `(x, y)` within `POS_EPS` (bulge ignored).
fn assert_vertex_at(pline: &Polyline, index: usize, x: f64, y: f64) {
    let v = pline.at(index);
    assert!(
        (v.x - x).abs() < POS_EPS && (v.y - y).abs() < POS_EPS,
        "vertex {index}: expected ({x}, {y}), got ({}, {})",
        v.x,
        v.y
    );
}

/// Open L-shape polyline with a single corner. Offset to the reflex side (`-2`) forces a
/// false-intersect extension at the corner, which is where Round/Bevel/Miter diverge.
fn l_shape_open() -> Polyline {
    let mut pl = Polyline::with_capacity(3, false);
    pl.add(0.0, 0.0, 0.0);
    pl.add(10.0, 0.0, 0.0);
    pl.add(10.0, 10.0, 0.0);
    pl
}

#[test]
fn default_options_are_round_and_miter_limit_4() {
    let opts = PlineOffsetOptions::<f64>::default();
    assert_eq!(opts.join_type, LineJoin::Round);
    assert!((opts.miter_limit - 4.0).abs() < 1e-9);
}

#[test]
fn miter_raw_offset_extends_to_miter_point() {
    // Miter point where the two offset lines (y = -2 and x = 12) meet is (12, -2). Miter length
    // is sqrt(8) ~= 2.83 which is well below the default limit (4 * 2 = 8), so no fallback.
    let pline = l_shape_open();
    let result = raw_offset(&pline, -2.0, LineJoin::Miter, 4.0);

    // (0,-2) -> miter point (12,-2) -> end (12,10). The (12,0) corner is collinear on x = 12 and
    // is absorbed, giving 3 vertices.
    assert_eq!(result.vertex_count(), 3);
    assert_vertex_at(&result, 0, 0.0, -2.0);
    assert_vertex_at(&result, 1, 12.0, -2.0);
    assert_vertex_at(&result, 2, 12.0, 10.0);
    // miter join is all straight lines (no arc bulge)
    assert!(result.iter_vertexes().all(|v| v.bulge == 0.0));
}

#[test]
fn bevel_raw_offset_connects_endpoints_with_line() {
    let pline = l_shape_open();
    let result = raw_offset(&pline, -2.0, LineJoin::Bevel, 4.0);

    // (0,-2) -> seg1 end (10,-2) -> seg2 start (12,0) -> end (12,10). Straight bevel line between
    // (10,-2) and (12,0), no arc.
    assert_eq!(result.vertex_count(), 4);
    assert_vertex_at(&result, 0, 0.0, -2.0);
    assert_vertex_at(&result, 1, 10.0, -2.0);
    assert_vertex_at(&result, 2, 12.0, 0.0);
    assert_vertex_at(&result, 3, 12.0, 10.0);
    assert!(result.iter_vertexes().all(|v| v.bulge == 0.0));
}

#[test]
fn round_raw_offset_uses_arc_at_corner() {
    let pline = l_shape_open();
    let result = raw_offset(&pline, -2.0, LineJoin::Round, 4.0);

    // Round adds an arc vertex at (10,-2) with a non-zero bulge, then (12,0), then end (12,10).
    assert_eq!(result.vertex_count(), 4);
    assert_vertex_at(&result, 0, 0.0, -2.0);
    assert_vertex_at(&result, 1, 10.0, -2.0);
    assert_vertex_at(&result, 2, 12.0, 0.0);
    assert_vertex_at(&result, 3, 12.0, 10.0);
    assert!(
        result.at(1).bulge.abs() > 1e-6,
        "round join should produce a non-zero arc bulge at the corner, got {}",
        result.at(1).bulge
    );
}

#[test]
fn miter_falls_back_to_bevel_when_limit_exceeded() {
    let pline = l_shape_open();

    let bevel = raw_offset(&pline, -2.0, LineJoin::Bevel, 4.0);
    // miter length (~2.83) exceeds 0.1 * 2 = 0.2, so miter must fall back to bevel geometry.
    let miter_clipped = raw_offset(&pline, -2.0, LineJoin::Miter, 0.1);

    assert_eq!(miter_clipped.vertex_count(), bevel.vertex_count());
    for i in 0..bevel.vertex_count() {
        let m = miter_clipped.at(i);
        let b = bevel.at(i);
        assert!(
            (m.x - b.x).abs() < POS_EPS && (m.y - b.y).abs() < POS_EPS && m.bulge == b.bulge,
            "miter-limit fallback vertex {i} differs from bevel: ({}, {}, {}) vs ({}, {}, {})",
            m.x,
            m.y,
            m.bulge,
            b.x,
            b.y,
            b.bulge
        );
    }
}

#[test]
fn miter_bevel_round_produce_distinct_geometry_at_reflex_corner() {
    let pline = l_shape_open();
    let miter = raw_offset(&pline, -2.0, LineJoin::Miter, 4.0);
    let bevel = raw_offset(&pline, -2.0, LineJoin::Bevel, 4.0);
    let round = raw_offset(&pline, -2.0, LineJoin::Round, 4.0);

    // miter collapses to 3 vertices; bevel and round keep 4 (but differ by bulge/path length).
    assert_ne!(miter.vertex_count(), bevel.vertex_count());
    assert_ne!(miter.vertex_count(), round.vertex_count());
    assert_ne!(
        round.at(1).bulge,
        bevel.at(1).bulge,
        "round and bevel must differ at the corner"
    );
}

#[test]
fn convex_corner_all_join_modes_coincide() {
    // Closed CCW square offset outward (+2). All four corners are convex 90°, so the offset
    // segments meet at a true intersect (the miter point) and Round/Miter/Bevel all trim to the
    // same point. Full parallel_offset results must be identical across join modes.
    let mut pline = Polyline::with_capacity(4, true);
    pline.add(0.0, 0.0, 0.0);
    pline.add(10.0, 0.0, 0.0);
    pline.add(10.0, 10.0, 0.0);
    pline.add(0.0, 10.0, 0.0);

    let round = pline.parallel_offset_opt(
        2.0,
        &PlineOffsetOptions {
            join_type: LineJoin::Round,
            ..Default::default()
        },
    );
    let miter = pline.parallel_offset_opt(
        2.0,
        &PlineOffsetOptions {
            join_type: LineJoin::Miter,
            ..Default::default()
        },
    );
    let bevel = pline.parallel_offset_opt(
        2.0,
        &PlineOffsetOptions {
            join_type: LineJoin::Bevel,
            ..Default::default()
        },
    );

    let to_props = |r: &[Polyline]| -> Vec<(usize, f64, f64)> {
        r.iter()
            .map(|p| (p.vertex_count(), p.area(), p.path_length()))
            .collect()
    };

    assert_eq!(round.len(), 1);
    assert_eq!(to_props(&round), to_props(&miter));
    assert_eq!(to_props(&round), to_props(&bevel));
}

#[test]
fn arc_corner_miter_differs_from_round_and_falls_back_to_bevel() {
    // Closed polyline whose first segment is an arc, exercising the arc-line / line-arc join
    // dispatch for miter/bevel. We don't hand-compute the miter point; instead we verify that
    // (a) miter with a generous limit differs from round, and (b) miter with a tiny limit falls
    // back to the same geometry as bevel.
    let mut pline = Polyline::with_capacity(4, true);
    pline.add(0.0, 0.0, 0.5);
    pline.add(10.0, 0.0, 0.0);
    pline.add(10.0, 10.0, 0.0);
    pline.add(0.0, 10.0, 0.0);

    let miter_big = raw_offset(&pline, -3.0, LineJoin::Miter, 1000.0);
    let miter_small = raw_offset(&pline, -3.0, LineJoin::Miter, 0.01);
    let bevel = raw_offset(&pline, -3.0, LineJoin::Bevel, 1000.0);
    let round = raw_offset(&pline, -3.0, LineJoin::Round, 1000.0);

    let key = |p: &Polyline| (p.vertex_count(), p.area(), p.path_length());

    // miter (unbounded) and round must differ somewhere on an arc-corner reflex offset.
    assert_ne!(
        key(&miter_big),
        key(&round),
        "miter and round should differ at an arc corner"
    );

    // miter with a tiny limit must collapse to the same geometry as bevel.
    assert_eq!(
        miter_small.vertex_count(),
        bevel.vertex_count(),
        "miter-limit fallback should match bevel vertex count at an arc corner"
    );
    assert!(
        (miter_small.area() - bevel.area()).abs() < 1e-4
            && (miter_small.path_length() - bevel.path_length()).abs() < 1e-4,
        "miter-limit fallback should match bevel area/path at an arc corner"
    );
}
