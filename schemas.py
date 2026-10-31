from typing import Literal, Union, Optional
from pydantic import BaseModel, Field


class Customer(BaseModel):
    role: Literal["customer"] = "customer"
    Distance: float = Field(..., ge=0, le=25)
    hour: Optional[int] = Field(None, ge=0, le=23)
    Weatherconditions: Literal["Sunny", "Stormy",
                               "Sandstorms", "Cloudy", "Fog", "Windy"]
    Road_traffic_density: Literal["Low", "Jam", "Medium", "High"]
    Type_of_vehicle: Literal["motorcycle",
                             "scooter", "electric_scooter", "bicycle"]
    Type_of_order: Literal["Snack", "Drinks", "Buffet", "Meal"]
    Festival: Literal["Yes", "No"]
    City: Literal["Urban", "Metropolitian", "Semi-Urban"]


class Rider(Customer):
    role: Literal["rider"] = "rider"
    Delivery_person_id: str
    Delivery_person_Age: int = Field(..., ge=17, le=60)
    # These three were missing: without them the rider's own rating, vehicle
    # condition and workload were ignored and replaced by defaults.
    Delivery_person_Ratings: float = Field(..., ge=1, le=5)
    Vehicle_condition: int = Field(..., ge=0, le=3)
    multiple_deliveries: int = Field(..., ge=0, le=3)


class PredictionDeliveryTime(BaseModel):
    estimated_time: int
    min_time: int
    max_time: int
    display_text: str


class DeliveryRequestWrapper(BaseModel):
    # The "role" field in the payload decides whether it's parsed as Customer or Rider
    user_perspective: Union[Customer, Rider] = Field(..., discriminator="role")
