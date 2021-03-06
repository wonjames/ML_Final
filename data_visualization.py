import numpy as np
import matplotlib.pyplot as plt
import pandas

spotify_df = pandas.read_csv('data.csv', header=0)

# Initialize histogram plots for numeric features
fig, axs = plt.subplots(2, 2, tight_layout=True)
num_bins = 20

# Add each feature to histogram plot
axs[0][0].hist(spotify_df['valence'], bins=num_bins)
axs[0][0].set_title('Valence')

axs[0][1].hist(spotify_df['acousticness'], bins=num_bins)
axs[0][1].set_title('Acousticness')

axs[1][0].hist(spotify_df['danceability'], bins=num_bins)
axs[1][0].set_title('Danceability')

axs[1][1].hist(spotify_df['duration_ms'] / 1000, bins=num_bins)
axs[1][1].set_title('Duration (seconds)')

plt.show()

fig, axs = plt.subplots(2, 2, tight_layout=True)
num_bins = 20

axs[0][0].hist(spotify_df['energy'], bins=num_bins)
axs[0][0].set_title('Energy')

axs[0][1].hist(spotify_df['explicit'], bins=2)
axs[0][1].set_title('Explicit (y/n)')
axs[0][1].set_xticks([0.25, 0.75])
axs[0][1].set_xticklabels(['no', 'yes'])

axs[1][0].hist(spotify_df['instrumentalness'], bins=num_bins)
axs[1][0].set_title('Instrumentalness')

axs[1][1].hist(spotify_df['key'], bins=12)
axs[1][1].set_title('Key')
axs[1][1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

plt.show()

fig, axs = plt.subplots(2, 2, tight_layout=True)
num_bins = 20

axs[0][0].hist(spotify_df['liveness'], bins=num_bins)
axs[0][0].set_title('Liveness')

axs[0][1].hist(spotify_df['loudness'], bins=num_bins)
axs[0][1].set_title('Loudness (dB)')

axs[1][0].hist(spotify_df['mode'], bins=2)
axs[1][0].set_title('Mode (minor/major)')
axs[1][0].set_xticks([0.25, 0.75])
axs[1][0].set_xticklabels(['minor', 'major'])

axs[1][1].hist(spotify_df['popularity'], bins=num_bins)
axs[1][1].set_title('Popularity')

plt.show()

fig, axs = plt.subplots(2, 2, tight_layout=True)
fig.delaxes(axs[1][1])
num_bins = 20

axs[0][0].hist(spotify_df['speechiness'], bins=num_bins)
axs[0][0].set_title('Speechiness')

axs[0][1].hist(spotify_df['tempo'], bins=num_bins)
axs[0][1].set_title('Tempo (BPM)')

axs[1][0].hist(spotify_df['year'], bins=num_bins)
axs[1][0].set_title('Release Year')

plt.show()

# Initialize a 4x4 plots for plotting against popularity
fig, axs = plt.subplots(2, 2, tight_layout=True)

axs[0][0].scatter(spotify_df['speechiness'], spotify_df['popularity'], color='#cc4365', s=1)
axs[0][0].set_xlabel('Speechiness')
axs[0][0].set_ylabel('Popularity')

axs[0][1].scatter(spotify_df['danceability'], spotify_df['popularity'], color='#cc4365', s=1)
axs[0][1].set_xlabel('Danceability')
axs[0][1].set_ylabel('Popularity')

axs[1][0].scatter(spotify_df['duration_ms'] / 1000, spotify_df['popularity'], color='#cc4365', s=1)
axs[1][0].set_xlabel('Duration (seconds)')
axs[1][0].set_ylabel('Popularity')

axs[1][1].scatter(spotify_df['year'], spotify_df['popularity'], color='#cc4365', s=1)
axs[1][1].set_xlabel('Year')
axs[1][1].set_ylabel('Popularity')

plt.show()