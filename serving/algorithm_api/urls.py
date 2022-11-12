from django.urls import path
from django.urls import include
from algorithm_api.views import post_one_algorithm
from algorithm_api.views import get_algorithm_docs



# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('ai/<slug:algorithm_name>', post_one_algorithm),
    path('ai/documents/<slug:algorithm_name>', get_algorithm_docs)
]

