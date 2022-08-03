# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
import streamlit as st
from path import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
import numpy as np
import seaborn as sns

# Load the data into a Pandas DataFrame
df_music_data = pd.read_csv(
    Path("Resources/spotify_dataset.csv"), index_col = 'uri')

#Copy data & Refine data
df_music_cp = df_music_data.copy()
## Get Rid of unnecessary data
df_music_filtered_list = df_music_cp.filter(['track', 'artist', 'uri','danceability','energy', 'key', 'loudness', 'mode', 'speechiness','acousticness','instrumentalness','liveness', 'valence', 'tempo', 'duration_ms', 'popularity', 'decade' ], axis=1)
df_music = df_music_filtered_list.drop(columns=['track', 'artist', 'decade'])


df = pd.get_dummies(df_music_filtered_list["decade"])
df  = pd.get_dummies(df_music_filtered_list["artist"])
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_music)
df_final = pd.concat([df_music, df])

df_final.fillna(0)

# Create a DataFrame with the scaled data
df_music_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_music.columns
)

# Copy the song URI names from the original data
df_music_data_scaled["uri"] = df_music_data.index

# Set the spotify URI column as index
df_music_data_scaled = df_music_data_scaled.set_index("uri")

# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1,11))


# Create an empy list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k : 
    model = KMeans(n_clusters= i, random_state =0)
    model.fit(df_music_data_scaled)
    inertia.append(model.inertia_)
    
# Create a dictionary with the data to plot the Elbow curve
elbow_data = { "k":k, "inertia":inertia}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_data = pd.DataFrame(elbow_data)


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow_plot = df_elbow_data.hvplot.line(x="k", y="inertia", title= "Find optimal k with elbow method for clusters for music reference data", xticks=k)
#df_elbow_plot

pca=PCA(n_components=6)
# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
pca_array = pca.fit_transform(df_music_data_scaled)

#View the array 
#pca_array[:6]

#Transform to dataframe
pca_df = pd.DataFrame(pca_array, columns= ["pca_1", "pca_2", "pca_3","pca_4", "pca_5", "pca_6"])

# View the first five rows of the DataFrame. 
#pca_df.head()

#pca.explained_variance_ratio_

# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=3)

# Fit the K-Means model using the scaled data
kmeans_model = model.fit(df_music_data_scaled )

# Predict the clusters to group the music using the scaled data
music_recommendation_cluster = model.predict(df_music_data_scaled)

# Create a copy of the DataFrame
df_music_data_scaled_copy = df_music_data_scaled.copy()

# Add a new column to the DataFrame with the predicted clusters
df_music_data_scaled_copy["music_cluster"] = music_recommendation_cluster

# distances 
distances = np.empty((0,len(df_music_data_scaled_copy.axes[0])), float)
cluster_centers = kmeans_model.cluster_centers_

# getting points and distances
for i, center_elem in enumerate(cluster_centers):
    # cdist is used to calculate the distance between center and other points
    distances = np.append(distances, cdist([center_elem],df_music_data_scaled[music_recommendation_cluster == i],'euclidean')) 
    

# Add a new column to the DataFrame with the distances from cluster centroid
df_music_data_scaled_copy["distances"] = distances

df_music_data_scaled_plot = df_music_data_scaled_copy.hvplot.scatter(x="danceability", y="loudness", by = "music_cluster")
#df_music_data_scaled_plot
    
#########Locate the favorite song 
def findSong(favsong):
    match_1_df = df_music_data[df_music_data['track'] == fav_song]
    fav_song_df =  df_music_data_scaled_copy[match_1_df.index.values.astype(str) == df_music_data_scaled_copy.index.values.astype(str)] 
    st.write(fav_song_df)
    return fav_song_df

#########Filter Logic
def filter_songs(fav_song_df, dancebility_threshold, energy_threshold):
    ##sorted_df = df_music_data_scaled_copy.sort_values(by=['music_cluster', 'distances']
    songs_filtered_by_cluster_df =     df_music_data_scaled_copy[df_music_data_scaled_copy['music_cluster'].values.astype(int) ==  fav_song_df.music_cluster.values.astype(int)]  
    
    ## setting up the tolerance to be 0.0001 . So, all songs within the limits described will be used.
    upper_distance_limit = fav_song_df.distances.values.astype(float) + 0.01
    lower_distance_limit = fav_song_df.distances.values.astype(float) - 0.01
    
    # filter based on distance 
    songs_filtered_by_distance_df = songs_filtered_by_cluster_df[(songs_filtered_by_cluster_df['distances'].values.astype(float) >  lower_distance_limit) & (songs_filtered_by_cluster_df['distances'].values.astype(float) <  upper_distance_limit)]
    
    ##set danceability band in comparison to their favorite song to refine recommendedation.
    upper_dance_limit = fav_song_df.danceability.values.astype(float) + dancebility_threshold
    lower_dance_limit = fav_song_df.distances.values.astype(float) - dancebility_threshold
    
    #filter based on danceability
    songs_filtered_by_danceability_df  = songs_filtered_by_distance_df [(songs_filtered_by_distance_df['danceability'].values.astype(float) > lower_dance_limit) & (songs_filtered_by_distance_df['danceability'].values.astype(float) < upper_dance_limit)]
    
    
    ##set danceability band in comparison to their favorite song to refine recommendedation.
    upper_energy_limit = fav_song_df.energy.values.astype(float) + energy_threshold
    lower_energy_limit = fav_song_df.energy.values.astype(float) - energy_threshold
    
    #filter based on energy
    songs_filtered_by_energy_df  = songs_filtered_by_danceability_df [(songs_filtered_by_danceability_df['energy'].values.astype(float) > lower_energy_limit) & (songs_filtered_by_danceability_df['energy'].values.astype(float) < upper_energy_limit)]
    
    
    ##Check if there are recommendations for filtering options choosen 
    try:
        recommended_songs_df = songs_filtered_by_energy_df.sample(5)
    except: 
        st.write("No recommendations from the data set, try different thresholds")
        
    playlist_df = pd.DataFrame()   
    #Return list of songs 
    for x in recommended_songs_df.index.tolist():
      playlist_df = playlist_df.append(df_music_filtered_list[str(x) == df_music_data.index.values.astype(str)], ignore_index = True)
    return playlist_df


############Streamlit Code #########################
st.title("Sensibilities")
st.subheader("Music recommendation App")
fav_song = st.selectbox('Pick your favorite song', df_music_data.sample(50).track)
dancebility_threshold= st.sidebar.slider("Dancebility Bandwidth",1, 3, 5)
energy_threshold = st.sidebar.slider("Energy Bandwidth",1, 3, 5,)

#Button to start recommendation process
if st.button('Recommend Playlist'):
    st.write(f'You have selected: { fav_song}')
    #Call the function to locate the favorite song
    fav_song_df = findSong(fav_song)
    #display the attributes of your favorite song
    st.write(f'Your favorite song belongs to cluster: {fav_song_df.music_cluster.values.astype(int)}')
    # Run through filters
    st.write( "Running through filters.... ")
    playlist = filter_songs(fav_song_df, dancebility_threshold , energy_threshold)
    #display the playlist
    st.write( "Here's your playlist . Hope you enjoy! ")
    st.write(playlist)






