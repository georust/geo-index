use geo_index::rtree::{sort::HilbertSort, NeighborsOptions, RTreeBuilder, RTreeIndex, RTreeRef};

#[test]
fn exact_distances_match_brute_force_after_sorting() {
    // Integer boxes, floating-point distances, loose envelopes, and enough items
    // to exercise multiple levels and insertion-index remapping.
    let points: Vec<_> = (0..97)
        .map(|i| ((i * 37 % 101) - 50, (i * 19 % 103) - 51))
        .collect();
    for node_size in [2, 8, 128] {
        let mut builder = RTreeBuilder::<i32>::new_with_node_size(points.len() as u32, node_size);
        for &(x, y) in &points {
            builder.add(x - 5, y - 7, x + 3, y + 2);
        }
        let tree = builder.finish::<HilbertSort>();
        let buffer = tree.into_inner();
        let tree = RTreeRef::<i32>::try_new(&buffer).unwrap();
        for query in [(0.3, -1.7), (100.1, 95.8)] {
            let distance = |id: usize| {
                let (x, y) = points[id];
                (x as f64 - query.0).abs() + 2.0 * (y as f64 - query.1).abs()
            };
            for limit in [None, Some(0), Some(1), Some(12), Some(200)] {
                for max_distance in [None, Some(0.0), Some(40.0), Some(distance(1))] {
                    let mut expected: Vec<_> = (0..points.len())
                        .filter(|id| id % 7 != 0)
                        .filter(|&id| max_distance.map_or(true, |max| distance(id) <= max))
                        .collect();
                    expected.sort_by(|&a, &b| distance(a).partial_cmp(&distance(b)).unwrap());
                    expected.truncate(limit.unwrap_or(usize::MAX));
                    let actual = tree.neighbors_with_callbacks(
                        NeighborsOptions {
                            k: limit,
                            max_distance,
                            include_tie_breakers: false,
                        },
                        |[min_x, min_y, max_x, max_y]| {
                            (query.0 - query.0.clamp(min_x as f64, max_x as f64)).abs()
                                + 2.0 * (query.1 - query.1.clamp(min_y as f64, max_y as f64)).abs()
                        },
                        |id, bbox| {
                            let (x, y) = points[id as usize];
                            assert_eq!(bbox, [x - 5, y - 7, x + 3, y + 2]);
                            (id % 7 != 0).then(|| distance(id as usize))
                        },
                    );
                    // Compare distances because tied items have no specified order.
                    assert_eq!(
                        actual
                            .iter()
                            .map(|&(id, dist)| {
                                assert_eq!(dist, distance(id as usize));
                                dist
                            })
                            .collect::<Vec<_>>(),
                        expected.iter().map(|&id| distance(id)).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn spherical_ranking_across_antimeridian_and_near_pole() {
    let points: [(f64, f64); 6] = [
        (179.9, 80.0),
        (-179.8, 80.0),
        (0.0, 89.9),
        (90.0, 85.0),
        (175.0, 75.0),
        (0.0, -80.0),
    ];
    let mut builder = RTreeBuilder::new_with_node_size(points.len() as u32, 2);
    for &(lon, lat) in &points {
        builder.add(lon, lat, lon, lat);
    }
    let tree = builder.finish::<HilbertSort>();
    for (lon, lat) in [(179.95_f64, 80.0_f64), (-90.0, 89.9)] {
        // Angular great-circle distance; the callback can use any radius or units.
        let distance = |id: usize| {
            let (x, y) = points[id];
            let a = ((y - lat).to_radians() / 2.0).sin().powi(2)
                + lat.to_radians().cos()
                    * y.to_radians().cos()
                    * ((x - lon).to_radians() / 2.0).sin().powi(2);
            2.0 * a.sqrt().asin()
        };
        let mut expected: Vec<_> = (0..points.len() as u32).collect();
        expected.sort_by(|&a, &b| {
            distance(a as usize)
                .partial_cmp(&distance(b as usize))
                .unwrap()
        });
        let actual = tree.neighbors_with_callbacks(
            NeighborsOptions::k(3),
            |_| 0.0,
            |id, _| Some(distance(id as usize)),
        );
        assert_eq!(
            actual,
            expected[..3]
                .iter()
                .map(|&id| (id, distance(id as usize)))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn limits_empty_ties_and_pruning() {
    let empty = RTreeBuilder::<f64>::new(0).finish::<HilbertSort>();
    assert!(empty
        .neighbors_with_callbacks::<f64, _, _>(
            NeighborsOptions::all(),
            |_| panic!("empty tree"),
            |_, _| panic!("empty tree")
        )
        .is_empty());

    let mut builder = RTreeBuilder::<f64>::new_with_node_size(64, 2);
    for x in 0..64 {
        builder.add(x as f64, 0.0, x as f64, 0.0);
    }
    let tree = builder.finish::<HilbertSort>();
    assert!(tree
        .neighbors_with_callbacks::<f64, _, _>(
            NeighborsOptions::k(0),
            |_| panic!("zero limit"),
            |_, _| panic!("zero limit")
        )
        .is_empty());
    assert!(tree.neighbors(0.0, 0.0, Some(0), None).is_empty());
    let mut calls = 0;
    let result = tree.neighbors_with_callbacks(
        NeighborsOptions::all().max_distance(1.0),
        |bbox| bbox[0],
        |_, bbox| {
            calls += 1;
            Some(bbox[0])
        },
    );
    assert_eq!(result, vec![(0, 0.0), (1, 1.0)]);
    assert!(calls < 64, "far nodes should be pruned");
    let mut ties =
        tree.neighbors_with_callbacks(NeighborsOptions::all(), |_| 0.0, |_, _| Some(1.0));
    ties.sort_unstable_by_key(|&(id, _)| id);
    assert_eq!(ties, (0..64).map(|id| (id, 1.0)).collect::<Vec<_>>());
    assert!(tree
        .neighbors_with_callbacks(NeighborsOptions::all(), |_| 0.0, |_, _| None)
        .is_empty());
}

#[test]
fn includes_ties_across_nodes_and_skips_excluded_items() {
    let mut builder = RTreeBuilder::<f64>::new_with_node_size(32, 2);
    for x in 0..32 {
        builder.add(x as f64, 0.0, x as f64, 0.0);
    }
    let tree = builder.finish::<HilbertSort>();
    for include_ties in [false, true] {
        let results = tree.neighbors_with_callbacks(
            NeighborsOptions::k(2).include_tie_breakers(include_ties),
            |_| 0.0,
            |id, _| {
                (id != 0).then_some(if id == 1 {
                    0.0
                } else if id % 2 == 0 {
                    1.0
                } else {
                    2.0
                })
            },
        );
        assert_eq!(results[0], (1, 0.0));
        assert_eq!(results.len(), if include_ties { 16 } else { 2 });
        assert!(results[1..]
            .iter()
            .all(|&(id, distance)| id % 2 == 0 && distance == 1.0));
    }
}
