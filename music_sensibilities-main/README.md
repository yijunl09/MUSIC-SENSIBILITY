# music_sensibility
To detect music taste profile of Spotify users and provide recommendations . 

--Implementation logic 
1. Cleanse the data retrieved from spotify 
2. Use elbow method to identify optimum clusters 
3. Use PCA technique, Perform Kmeans clustering . Per data collcted, the optimal clusters were 3.
4. Request the user to enter their favorite song from a sample of 30 listed in the drop down 
5. Locate the favorite song in the clusters 
6. Calculate the  distance from the centeriod for all nodes 
7. Filter nodes closest to the favorite song located 
8. Set distance Tolerance as 0.001 and filter to identify nodes/points/tracks closest to the favorite song.
9. Filter by dancebility bandwidth 
10. Filter by Energy bandwidth 
11. Display playlist recommendation by sending a sample froom the filtered list. 

-- Instructions to run
streamlit run music_sensibility.py 

