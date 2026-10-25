
from django.urls import path

from .views import (
    predict_late_delivery,
    forecast_demand,
    list_product_models,
)

urlpatterns = [
    path("predict/", predict_late_delivery, name="predict_late_delivery"),
    path("forecast/", forecast_demand, name="forecast_demand"),
    path("product-models/", list_product_models, name="list_product_models"),
]
