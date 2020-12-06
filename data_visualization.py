import numpy as np
import matplotlib.pyplot as plt
import pandas

spotify_df = pandas.read_csv('data.csv', header=0, nrows=10000)
print(spotify_df['valence'])

# Initialize 16 histogram plots for numeric features
fig, axs = plt.subplots(4, 4, figsize=(12, 10), tight_layout=True)
num_bins = 20

# Add each feature to histogram plot
axs[0][0].hist(spotify_df['valence'], bins=num_bins)
axs[0][0].set_title('Valence')

axs[0][1].hist(spotify_df['acousticness'], bins=num_bins)
axs[0][1].set_title('Acousticness')

axs[0][2].hist(spotify_df['danceability'], bins=num_bins)
axs[0][2].set_title('Danceability')

axs[0][3].hist(spotify_df['duration_ms'], bins=num_bins)
axs[0][3].set_title('Duration (ms)')

axs[1][0].hist(spotify_df['energy'], bins=num_bins)
axs[1][0].set_title('Energy')

axs[1][1].hist(spotify_df['explicit'], bins=2)
axs[1][1].set_title('Explicit (y/n)')
axs[1][1].set_xticks([0.25, 0.75])
axs[1][1].set_xticklabels(['no', 'yes'])

axs[1][2].hist(spotify_df['instrumentalness'], bins=num_bins)
axs[1][2].set_title('Instrumentalness')

axs[1][3].hist(spotify_df['key'], bins=12)
axs[1][3].set_title('Key')
axs[1][3].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

axs[2][0].hist(spotify_df['liveness'], bins=num_bins)
axs[2][0].set_title('Liveness')

axs[2][1].hist(spotify_df['loudness'], bins=num_bins)
axs[2][1].set_title('Loudness (dB)')

axs[2][2].hist(spotify_df['mode'], bins=2)
axs[2][2].set_title('Mode (minor/major)')
axs[2][2].set_xticks([0.25, 0.75])
axs[2][2].set_xticklabels(['minor', 'major'])

axs[2][3].hist(spotify_df['popularity'], bins=num_bins)
axs[2][3].set_title('Popularity')

axs[3][0].hist(spotify_df['speechiness'], bins=num_bins)
axs[3][0].set_title('Speechiness')

axs[3][1].hist(spotify_df['tempo'], bins=num_bins)
axs[3][1].set_title('Tempo')

axs[3][2].hist(spotify_df['year'], bins=num_bins)
axs[3][2].set_title('Release Year')

plt.show()