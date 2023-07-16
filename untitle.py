import matplotlib
import numpy as np 


rgb_colors = {}
for name, hex in matplotlib.colors.cnames.items():
    rgb_colors[name] = list([int(np.round(i * 255)) for i in matplotlib.colors.to_rgb(hex)])
    if len(rgb_colors) == 20:
        break

print(rgb_colors)

import yaml


with open('config/rgb_colors.yml', 'w') as outfile:
    yaml.dump(rgb_colors, outfile, default_flow_style=Truea)


