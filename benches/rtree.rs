use bytemuck::cast_slice;
use criterion::{criterion_group, criterion_main, Criterion};
use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::util::f64_box_to_f32;
use geo_index::rtree::{RTree, RTreeBuilder, RTreeIndex};
use geo_index::IndexableNum;
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::AABB;
use std::fs::read;

fn load_data() -> Vec<f64> {
    let buf = read("benches/bench_data/bounds.raw").unwrap();
    cast_slice(&buf).to_vec()
}

fn construct_rtree<N: IndexableNum>(boxes_buf: &[N]) -> RTree<N> {
    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<HilbertSort>()
}

fn construct_rtree_str<N: IndexableNum>(boxes_buf: &[N]) -> RTree<N> {
    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<STRSort>()
}

fn construct_rtree_f32_with_cast(boxes_buf: &[f64]) -> RTree<f32> {
    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        let (min_x, min_y, max_x, max_y) = f64_box_to_f32(min_x, min_y, max_x, max_y);
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<HilbertSort>()
}

fn construct_rstar(
    rect_vec: Vec<GeomWithData<Rectangle<(f64, f64)>, usize>>,
) -> rstar::RTree<GeomWithData<Rectangle<(f64, f64)>, usize>> {
    rstar::RTree::bulk_load(rect_vec)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let boxes_buf = load_data();
    let boxes_f32_buf = boxes_buf.iter().map(|x| *x as f32).collect::<Vec<_>>();
    let aabb_vec: Vec<AABB<(f64, f64)>> = boxes_buf
        .chunks(4)
        .map(|box_| AABB::from_corners((box_[0], box_[1]), (box_[2], box_[3])))
        .collect();
    let rect_vec: Vec<GeomWithData<Rectangle<_>, usize>> = aabb_vec
        .into_iter()
        .enumerate()
        .map(|(idx, aabb)| GeomWithData::new(aabb.into(), idx))
        .collect();

    c.bench_function("construction (geo-index, hilbert)", |b| {
        b.iter(|| construct_rtree(&boxes_buf))
    });

    c.bench_function("construction (geo-index, STRTree)", |b| {
        b.iter(|| construct_rtree_str(&boxes_buf))
    });

    c.bench_function(
        "construction (geo-index, hilbert, f64 to f32, including casting)",
        |b| b.iter(|| construct_rtree_f32_with_cast(&boxes_buf)),
    );

    c.bench_function("construction (geo-index f32)", |b| {
        b.iter(|| construct_rtree(&boxes_f32_buf))
    });

    c.bench_function("construction (rstar bulk)", |b| {
        b.iter(|| construct_rstar(rect_vec.to_vec()))
    });

    let flatbush_tree = construct_rtree(&boxes_buf);
    let flatbush_str_tree = construct_rtree_str(&boxes_buf);
    let flatbush_f32_tree = construct_rtree_f32_with_cast(&boxes_buf);
    let rstar_tree = construct_rstar(rect_vec.to_vec());
    let (min_x, min_y, max_x, max_y) = (-112.007_493, 40.633_799, -111.920_964, 40.694_228);
    let min_x_f32 = min_x as f32;
    let min_y_f32 = min_y as f32;
    let max_x_f32 = max_x as f32;
    let max_y_f32 = max_y as f32;

    let flatbush_search_results = flatbush_tree.search(min_x, min_y, max_x, max_y);
    let flatbush_str_search_results = flatbush_str_tree.search(min_x, min_y, max_x, max_y);
    let flatbush_f32_search_results =
        flatbush_f32_tree.search(min_x_f32, min_y_f32, max_x_f32, max_y_f32);
    let rstar_search_results = {
        let aabb = AABB::from_corners((min_x, min_y), (max_x, max_y));
        rstar_tree
            .locate_in_envelope_intersecting(&aabb)
            .collect::<Vec<_>>()
    };

    assert_eq!(flatbush_search_results.len(), rstar_search_results.len());
    println!(
        "search() results in {} items",
        flatbush_search_results.len()
    );
    println!(
        "search() on STRTree results in {} items",
        flatbush_str_search_results.len()
    );
    println!(
        "search() on f32 results in {} items",
        flatbush_f32_search_results.len()
    );
    println!(
        "flatbush buffer size: {} bytes",
        flatbush_tree.clone().into_inner().len()
    );
    println!(
        "flatbush f32 buffer size: {} bytes",
        flatbush_f32_tree.clone().into_inner().len()
    );

    c.bench_function("search (flatbush)", |b| {
        b.iter(|| flatbush_tree.search(min_x, min_y, max_x, max_y))
    });

    c.bench_function("search (flatbush STRTree)", |b| {
        b.iter(|| flatbush_str_tree.search(min_x, min_y, max_x, max_y))
    });

    c.bench_function("search (flatbush f32)", |b| {
        b.iter(|| flatbush_f32_tree.search(min_x_f32, min_y_f32, max_x_f32, max_y_f32))
    });

    c.bench_function("search (rstar)", |b| {
        b.iter(|| {
            let aabb = AABB::from_corners((min_x, min_y), (max_x, max_y));
            rstar_tree
                .locate_in_envelope_intersecting(&aabb)
                .collect::<Vec<_>>()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
