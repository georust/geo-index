import geopandas as gpd
import shapely


gdf = gpd.read_file("./bench_data/Utah.geojson", engine="pyogrio")
bounds = shapely.bounds(gdf.geometry)
print(bounds.shape)
buf = bounds.tobytes("C")
with open("./bench_data/bounds.raw", "wb") as f:
    f.write(buf)
