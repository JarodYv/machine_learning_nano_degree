# Lesson 3.4: Make Classes
# Mini-Project: Movies Website

# In this file, you will define the class Movie. You could do this
# directly in entertainment_center.py but many developers keep their
# class definitions separate from the rest of their code. This also
# gives you practice importing Python files.

import webbrowser
import re

class Movie():
    # This class provides a way to store movie related information

    def __init__(self, name, rate, poster, trailer_url, IMDb, director, star_list):
        # initialize instance of class Movie
        self.name = name 			# the name of the film
        self.rate = rate 			# the rate score from IMDb, a float from 0~10
        self.poster = poster 		# the poster url of the film
        self.trailer_url = trailer_url# the trailer url of the film on Youtube
        self.IMDb = IMDb 			# the IMDb NO. of the film
        self.director = director 	# the director of the film
        self.star_list = star_list	# a list of stars in the film

    def get_stars(self):
        # join star list to a string with star names separated by comma
        stars = ''
        for star in self.star_list:
            stars += star
            stars += ', '
        return stars[:len(stars)-2]

    def get_trailer_youtube_id(self):
        # Extract the youtube ID from the url
        youtube_id_match = re.search(r'(?<=v=)[^&#]+', self.trailer_url)
        youtube_id_match = youtube_id_match or re.search(r'(?<=be/)[^&#]+', self.trailer_url)
        trailer_youtube_id = youtube_id_match.group(0) if youtube_id_match else None
        return trailer_youtube_id