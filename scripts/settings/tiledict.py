import folium 

folium_layers = {
        'Stamen TonerLite': folium.TileLayer(
                tiles = 'https://stamen-tiles-{s}.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}{r}.png',
                attr = 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                name = 'Stamen TonerLite',
                control = True,
                overlay = True,
                show = True
        ), 
        'Google Satellite': folium.TileLayer(
                tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr = 'Google',
                name = 'Google Satellite',
                overlay = True,
                control = True,
                show = False
                ),
        'CyclOSM': folium.TileLayer(
                tiles = 'https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png',
                attr = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                name = 'CyclOSM',
                control = True,
                overlay = True,
                show = False
        ),     
        'OSM': folium.TileLayer(
                tiles = 'openstreetmap', 
                name = 'OpenStreetMap',
                attr = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                control = True, 
                overlay = True,
                show = False
                )
}