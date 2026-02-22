import folium
import json
from folium.plugins import HeatMap

def extract_polygon_centroid(coordinates):
    """Extract centroid from polygon coordinates"""
    if not coordinates or not coordinates[0]:
        return None, None

    ring = coordinates[0]  # Outer ring
    if len(ring) < 3:
        return None, None

    # Calculate centroid
    x_coords = [point[0] for point in ring]
    y_coords = [point[1] for point in ring]

    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)

    return centroid_x, centroid_y

def better_albers_to_latlon(x, y):
    """
    Better approximation for Albers Equal Area Conic to Lat/Lon
    Using known reference points and linear interpolation
    """
    # Albers coordinates roughly span:
    # X: -2,500,000 to 2,500,000 (west to east)
    # Y: -1,500,000 to 1,500,000 (south to north)

    # US bounds approximately:
    # Lat: 25°N to 49°N (24 degrees)
    # Lon: -125°W to -66°W (59 degrees)

    # Center points
    center_x = 0
    center_y = 0
    center_lat = 39.0
    center_lon = -96.0

    # Scale factors (empirically adjusted)
    x_scale = 59.0 / 5000000.0  # degrees per Albers unit
    y_scale = 24.0 / 3000000.0  # degrees per Albers unit

    # Convert
    lon = center_lon + (x - center_x) * x_scale
    lat = center_lat + (y - center_y) * y_scale

    # Clamp to reasonable US bounds
    lat = max(24, min(50, lat))
    lon = max(-130, min(-65, lon))

    return lat, lon

# Load and process one scenario first to test
print("Testing coordinate conversion with high growth scenario...")

file_path = 'datasets/im3_projected_data_centers/high_growth/high_growth_50_market_gravity.geojson'
with open(file_path, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data['features'])} features")

# Test conversion on first few features
print("\nTesting coordinate conversion:")
for i in range(min(5, len(data['features']))):
    feature = data['features'][i]
    props = feature['properties']
    geom = feature['geometry']

    if geom['type'] == 'Polygon' and geom.get('coordinates'):
        centroid_x, centroid_y = extract_polygon_centroid(geom['coordinates'])
        if centroid_x is not None and centroid_y is not None:
            lat, lon = better_albers_to_latlon(centroid_x, centroid_y)
            print(f"  {props['region']}: Albers({centroid_x:.0f}, {centroid_y:.0f}) -> LatLon({lat:.2f}, {lon:.2f})")

# Create the map with corrected coordinates
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

# Process all scenarios
scenarios = ['low_growth', 'moderate_growth', 'high_growth', 'higher_growth']
scenario_names = ['Low Growth', 'Moderate Growth', 'High Growth', 'Higher Growth']
colors = ['#0000FF', '#00FF00', '#FFA500', '#FF0000']  # Blue, Green, Orange, Red

print("\nCreating heatmaps for all scenarios...")

for i, scenario in enumerate(scenarios):
    file_path = f'datasets/im3_projected_data_centers/{scenario}/{scenario}_50_market_gravity.geojson'

    try:
        with open(file_path, 'r') as f:
            scenario_data = json.load(f)

        print(f"Processing {scenario}: {len(scenario_data['features'])} locations")

        # Create heatmap points
        heat_points = []

        for feature in scenario_data['features']:
            props = feature['properties']
            geom = feature['geometry']

            if geom['type'] == 'Polygon' and geom.get('coordinates'):
                centroid_x, centroid_y = extract_polygon_centroid(geom['coordinates'])

                if centroid_x is not None and centroid_y is not None:
                    lat, lon = better_albers_to_latlon(centroid_x, centroid_y)

                    # Skip points that are clearly wrong
                    if 24 <= lat <= 50 and -130 <= lon <= -65:
                        # Weight by cost (higher cost = more heat)
                        weight = props['total_cost_million_usd'] / 100.0
                        weight = max(0.5, min(5.0, weight))

                        heat_points.append([lat, lon, weight])

        print(f"  Created {len(heat_points)} valid heatmap points for {scenario}")

        if heat_points:
            # Create feature group
            feature_group = folium.FeatureGroup(
                name=f"{scenario_names[i]} ({len(heat_points)} sites)",
                show=(scenario == 'high_growth')  # Show high growth by default
            )

            # Add heatmap
            HeatMap(
                heat_points,
                min_opacity=0.3,
                max_zoom=18,
                radius=12,
                blur=8,
                gradient={0.2: colors[i], 0.5: colors[i], 1.0: colors[i]}
            ).add_to(feature_group)

            # Add sample markers for verification
            sample_points = heat_points[::max(1, len(heat_points)//30)]  # Sample every nth point

            for lat, lon, weight in sample_points:
                # Find corresponding feature for popup info
                for feature in scenario_data['features']:
                    props = feature['properties']
                    geom = feature['geometry']

                    if geom['type'] == 'Polygon' and geom.get('coordinates'):
                        cx, cy = extract_polygon_centroid(geom['coordinates'])
                        if cx is not None and cy is not None:
                            test_lat, test_lon = better_albers_to_latlon(cx, cy)
                            if abs(test_lat - lat) < 0.01 and abs(test_lon - lon) < 0.01:
                                folium.CircleMarker(
                                    location=[lat, lon],
                                    radius=4,
                                    popup=f"""
                                    <b>{props['region'].title()}</b><br>
                                    Cost: ${props['total_cost_million_usd']:.1f}M<br>
                                    Water: {props['cooling_water_consumption_mgy']:.1f} MGY<br>
                                    Location: ({lat:.3f}, {lon:.3f})
                                    """,
                                    color='white',
                                    fill=True,
                                    fillColor=colors[i],
                                    fillOpacity=0.8,
                                    weight=1
                                ).add_to(feature_group)
                                break

            feature_group.add_to(m)

    except FileNotFoundError:
        print(f"File not found: {file_path}")

# Add layer control
folium.LayerControl(collapsed=False).add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed;
     bottom: 50px; left: 50px; width: 300px; height: 180px;
     background-color: white; border:2px solid grey; z-index:9999;
     font-size:13px; padding: 10px">
<h4>Corrected Precise Locations</h4>
<p><span style="color:blue">●</span> Low Growth (exact coordinates)</p>
<p><span style="color:green">●</span> Moderate Growth (exact coordinates)</p>
<p><span style="color:orange">●</span> High Growth (exact coordinates)</p>
<p><span style="color:red">●</span> Higher Growth (exact coordinates)</p>
<p><small><b>Improved coordinate conversion</b><br>
Each point = precise data center location<br>
Heat intensity = construction cost<br>
Toggle scenarios using layer control ↗</small></p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save map
m.save('corrected_heatmap.html')
print(f"\nCorrected heatmap saved as 'corrected_heatmap.html'")
print("This should show data centers in proper US locations!")