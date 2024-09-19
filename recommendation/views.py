from django.shortcuts import render
from .ml_model import load_data, calculate_similarity, recommend

# Load the dataset and calculate similarity (you can cache this for performance)
new = load_data()
similarity = calculate_similarity(new)



def index(request):
    recommendations = None
    message = None
    if request.method == 'POST':
        movie = request.POST.get('movie')
        if movie:
            recommendations = recommend(movie, new, similarity)
            if recommendations is None:
                message = f"Sorry, the movie '{movie}' is not in the database."
    
    return render(request, 'recommendation/index.html', {'recommendations': recommendations, 'message': message})
