# Lesson 3.4: Make Classes
# Mini-Project: Movies Website

# In this file, you will define instances of the class Movie defined
# in media.py. After you follow along with Kunal, make some instances
# of your own!

# After you run this code, open the file fresh_tomatoes.html to
# see your webpage!

import media
import fresh_tomatoes

the_invisible_guest = media.Movie(
	"The Invisible Guest", 
	8.1, 
	"https://images-na.ssl-images-amazon.com/images/M/MV5BOTEwYTkzMTMtODEzNC00YWM2LTgxNDMtZWNkNTQzNDVjMWQ2L2ltYWdlL2ltYWdlXkEyXkFqcGdeQXVyMDY4NTQ1MA@@._V1_UY268_CR3,0,182,268_AL_.jpg",
	"https://www.youtube.com/watch?v=vwyZH85NQC4",
	"tt4857264",
	"Oriol Paulo",
	["Mario Casas", "Ana Wagener", "Jose Coronado"]
	)

thor_3 = media.Movie(
	"Thor: Ragnarok", 
	8.2, 
	"https://images-na.ssl-images-amazon.com/images/M/MV5BMjMyNDkzMzI1OF5BMl5BanBnXkFtZTgwODcxODg5MjI@._V1_UX182_CR0,0,182,268_AL_.jpg",
	"https://www.youtube.com/watch?v=vwyZH85NQC4",
	"tt3501632",
	"Taika Waititi",
	["Chris Hemsworth", "Tom Hiddleston", "Cate Blanchett"]
	)

kingsman_2 = media.Movie(
	"Kingsman 2", 
	7.2, 
	"https://images-na.ssl-images-amazon.com/images/M/MV5BMjQ3OTgzMzY4NF5BMl5BanBnXkFtZTgwOTc4OTQyMzI@._V1_UX182_CR0,0,182,268_AL_.jpg",
	"https://www.youtube.com/watch?v=vwyZH85NQC4",
	"tt4649466",
	"Matthew Vaughn",
	["Taron Egerton", "Colin Firth", "Mark Strong"]
	)

blade_runner_2049 = media.Movie(
	"Blade Runner 2049", 
	8.5, 
	"https://images-na.ssl-images-amazon.com/images/M/MV5BNzA1Njg4NzYxOV5BMl5BanBnXkFtZTgwODk5NjU3MzI@._V1_UX182_CR0,0,182,268_AL_.jpg",
	"https://www.youtube.com/watch?v=vwyZH85NQC4",
	"tt1856101",
	"Denis Villeneuve",
	["Harrison Ford", "Ryan Gosling", "Ana de Armas"]
	)

geostorm = media.Movie(
	"Geostorm", 
	5.7, 
	"https://images-na.ssl-images-amazon.com/images/M/MV5BMTA0OTQwMTIxNzheQTJeQWpwZ15BbWU4MDQ1MzI3OTMy._V1_UX182_CR0,0,182,268_AL_.jpg",
	"https://www.youtube.com/watch?v=vwyZH85NQC4",
	"tt1981128",
	"Dean Devlin",
	["Gerard Butler", "Jim Sturgess", "Abbie Cornish"]
	)

murder_on_the_orient_express = media.Movie(
	"Murder on the Orient Express", 
	6.8, 
	"https://images-na.ssl-images-amazon.com/images/M/MV5BNGFmM2NmYjYtMjAwNy00ZDkzLWI3ZWMtOGZhOTRhYzQwMTA0XkEyXkFqcGdeQXVyNzU2MzMyNTI@._V1_UX182_CR0,0,182,268_AL_.jpg",
	"https://www.youtube.com/watch?v=vwyZH85NQC4",
	"tt3402236",
	"Kenneth Branagh",
	["Kenneth Branagh", "Penelope Cruz", "Willem Dafoe"]
	)

movies = [the_invisible_guest, thor_3, kingsman_2, blade_runner_2049, geostorm, murder_on_the_orient_express]
fresh_tomatoes.open_movies_page(movies)
