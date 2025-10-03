"""
Streamlit App for Airbnb Price Prediction - Manchester UK
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
import preprocessing

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    """Load the trained model, scaler, feature columns, and defaults"""
    try:
        model = joblib.load('original_airbnb_model.pkl')
        scaler = joblib.load('original_scaler.pkl')
        feature_columns = joblib.load('original_feature_columns.pkl')
        defaults = joblib.load('feature_defaults.pkl')
        return model, scaler, feature_columns, defaults
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None, None

def main():
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF5A5F;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 0.75rem;
            border-radius: 8px;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>🏠 Airbnb Price Predictor - Manchester</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Enter your property details to instantly get a personalised price prediction based on similar Airbnb listings in your area.</p>", unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_columns, defaults = load_model()
    
    if model is None:
        st.error("Model files not found. Please ensure all .pkl files are in the folder.")
        return
    
    st.markdown("---")
    
    # Two column layout
    left_col, right_col = st.columns(2)
    
    with left_col:
        # Basic Information
        st.subheader("Basic Information")
        name = st.text_input("Listing Name", "Cozy City Centre Apartment")
        description = st.text_area(
            "Description", 
            "Beautiful modern apartment in the heart of Manchester. Close to all amenities, perfect for business or leisure.",
            height=100
        )
        picture_url = st.text_input(
            "Picture URL", 
            "https://a0.muscache.com/pictures/12345/example_original.jpg"
        )
        
        st.markdown("---")
        
        # Property Details
        st.subheader("Property Details")
        
        property_type = st.selectbox("Property Type", [
            "Entire home", "Entire condo", "Private room", "Entire rental unit",
            "Entire serviced apartment", "Entire townhouse", "Other"
        ])
        
        room_type = st.selectbox("Room Type", [
            "Entire home/apt", "Private room", "Shared room", "Hotel room"
        ])
        
        col_a, col_b = st.columns(2)
        with col_a:
            accommodates = st.number_input("Accommodates", 1, 16, 2)
            beds = st.number_input("Beds", 0, 20, 1)
        
        with col_b:
            bedrooms = st.number_input("Bedrooms", 0, 10, 1)
            bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0, 0.5)
        
        st.markdown("---")
        
        # Location
        st.subheader("Location")
        
        neighbourhood_options = [
            'City Centre', 'Trafford District', 'Bury District', 'Bolton District',
            'Salford District', 'Stockport District', 'Tameside District',
            'Rochdale District', 'Oldham District', 'Wigan District',
            'Harpurhey', 'Longsight', 'Hulme', 'Old Moat', 'Fallowfield',
            'Whalley Range', 'Levenshulme', 'Didsbury West', 'Crumpsall',
            'Moss Side', 'Bradford', 'Miles Platting and Newton Heath',
            'Rusholme', 'Withington', 'Gorton South', 'Chorlton Park',
            'Chorlton', 'Cheetham', 'Ardwick', 'Gorton North',
            'Northenden', 'Woodhouse Park', 'Didsbury East'
        ]
        
        neighbourhood_cleansed = st.selectbox(
            "Neighbourhood",
            sorted(neighbourhood_options),
            index=sorted(neighbourhood_options).index('City Centre')
        )
        
        neighbourhood_coords = {
            'City Centre': (53.4808, -2.2426),
            'Bolton District': (53.5768, -2.4282),
            'Bury District': (53.5933, -2.2958),
            'Salford District': (53.4875, -2.2901),
            'Stockport District': (53.4106, -2.1575),
            'Trafford District': (53.4233, -2.3533),
            'Rochdale District': (53.6097, -2.1561),
            'Oldham District': (53.5409, -2.1114),
            'Tameside District': (53.4804, -2.0809),
            'Wigan District': (53.5450, -2.6318),
        }
        
        default_lat, default_lng = neighbourhood_coords.get(neighbourhood_cleansed, (53.4808, -2.2426))
        
        col_e, col_f = st.columns(2)
        with col_e:
            latitude = st.number_input("Latitude", value=default_lat, format="%.4f")
        with col_f:
            longitude = st.number_input("Longitude", value=default_lng, format="%.4f")
        
        st.markdown("---")
        
        # Host Information
        st.subheader("Host Information")
        
        host_since = st.date_input("Host Since", datetime(2020, 1, 1))
        host_total_listings_count = st.number_input("Total Listings", min_value=1, max_value=100, value=1)
        host_response_time = st.selectbox("Response Time", [
            "within an hour", "within a few hours", "within a day", "a few days or more"
        ])
        
        col_i, col_j, col_k = st.columns(3)
        with col_i:
            host_is_superhost = st.checkbox("Superhost")
        with col_j:
            host_identity_verified = st.checkbox("Identity Verified", value=True)
        with col_k:
            instant_bookable = st.checkbox("Instant Bookable")
    
    with right_col:
        # Amenities with checkboxes
        st.subheader("Amenities")
        st.write("Select all amenities your property offers:")
        
        amenity_col1, amenity_col2 = st.columns(2)
        
        amenities_selected = []
        
        with amenity_col1:
            st.write("**Essentials**")
            if st.checkbox("WiFi", value=True):
                amenities_selected.append("Wifi")
            if st.checkbox("Kitchen", value=True):
                amenities_selected.append("Kitchen")
            if st.checkbox("TV", value=True):
                amenities_selected.append("TV")
            if st.checkbox("Heating", value=True):
                amenities_selected.append("Heating")
            if st.checkbox("Air Conditioning"):
                amenities_selected.append("Air conditioning")
            if st.checkbox("Essentials", value=True):
                amenities_selected.append("Essentials")
            
            st.write("**Comfort**")
            if st.checkbox("Washer"):
                amenities_selected.append("Washer")
            if st.checkbox("Dryer"):
                amenities_selected.append("Dryer")
            if st.checkbox("Hair Dryer"):
                amenities_selected.append("Hair dryer")
            if st.checkbox("Iron"):
                amenities_selected.append("Iron")
            if st.checkbox("Hangers"):
                amenities_selected.append("Hangers")
            if st.checkbox("Shampoo"):
                amenities_selected.append("Shampoo")
        
        with amenity_col2:
            st.write("**Premium**")
            if st.checkbox("Free Parking"):
                amenities_selected.append("Free parking")
            if st.checkbox("Workspace"):
                amenities_selected.append("Dedicated workspace")
            if st.checkbox("Pool"):
                amenities_selected.append("Pool")
            if st.checkbox("Hot Tub"):
                amenities_selected.append("Hot tub")
            if st.checkbox("Gym"):
                amenities_selected.append("Gym")
            if st.checkbox("Breakfast"):
                amenities_selected.append("Breakfast")
            
            st.write("**Safety & Access**")
            if st.checkbox("Self Check-in"):
                amenities_selected.append("Self check-in")
            if st.checkbox("Private Entrance"):
                amenities_selected.append("Private entrance")
            if st.checkbox("Lockbox"):
                amenities_selected.append("Lockbox")
            if st.checkbox("Smoke Alarm"):
                amenities_selected.append("Smoke alarm")
            if st.checkbox("Carbon Monoxide Alarm"):
                amenities_selected.append("Carbon monoxide alarm")
            if st.checkbox("Fire Extinguisher"):
                amenities_selected.append("Fire extinguisher")
        
        # Additional amenities
        with st.expander("More Amenities"):
            more_col1, more_col2 = st.columns(2)
            
            with more_col1:
                if st.checkbox("Balcony"):
                    amenities_selected.append("Balcony")
                if st.checkbox("Garden"):
                    amenities_selected.append("Garden")
                if st.checkbox("BBQ Grill"):
                    amenities_selected.append("BBQ grill")
                if st.checkbox("Dishwasher"):
                    amenities_selected.append("Dishwasher")
            
            with more_col2:
                if st.checkbox("Coffee Maker"):
                    amenities_selected.append("Coffee maker")
                if st.checkbox("Microwave"):
                    amenities_selected.append("Microwave")
                if st.checkbox("Refrigerator"):
                    amenities_selected.append("Refrigerator")
                if st.checkbox("Oven"):
                    amenities_selected.append("Oven")
        
        # Convert selected amenities to comma-separated string
        amenities = ", ".join(amenities_selected)
        
        st.markdown("---")
        
        # Reviews
        st.subheader("Reviews")
        
        number_of_reviews = st.number_input("Number of Reviews", 0, 1000, 5)
        
        st.write("Review Scores (1.0 - 5.0)")
        
        review_scores_rating = st.slider("Overall Rating", 1.0, 5.0, 4.5, 0.1)
        review_scores_cleanliness = st.slider("Cleanliness", 1.0, 5.0, 4.5, 0.1)
        review_scores_accuracy = st.slider("Accuracy", 1.0, 5.0, 4.5, 0.1)
        review_scores_checkin = st.slider("Check-in", 1.0, 5.0, 4.5, 0.1)
        review_scores_communication = st.slider("Communication", 1.0, 5.0, 4.5, 0.1)
        review_scores_location = st.slider("Location", 1.0, 5.0, 4.5, 0.1)
        review_scores_value = st.slider("Value", 1.0, 5.0, 4.5, 0.1)
    
    # Predict button
    st.markdown("---")
    if st.button("Get Price Prediction"):
        try:
            # Gather user data
            user_data = {
                'name': name,
                'description': description,
                'picture_url': picture_url,
                'property_type': property_type,
                'room_type': room_type,
                'accommodates': accommodates,
                'bathrooms': bathrooms,
                'bedrooms': bedrooms,
                'beds': beds,
                'amenities': amenities,
                'number_of_reviews': number_of_reviews,
                'review_scores_rating': review_scores_rating,
                'review_scores_cleanliness': review_scores_cleanliness,
                'review_scores_location': review_scores_location,
                'host_since': host_since,
                'host_response_time': host_response_time,
                'host_is_superhost': host_is_superhost,
                'host_total_listings_count': host_total_listings_count,
                'host_identity_verified': host_identity_verified,
                'neighbourhood_cleansed': neighbourhood_cleansed,
                'review_scores_accuracy': review_scores_accuracy,
                'review_scores_checkin': review_scores_checkin,
                'review_scores_communication': review_scores_communication,
                'review_scores_value': review_scores_value,
                'latitude': latitude,
                'longitude': longitude,
                'instant_bookable': instant_bookable
            }
            
            # Preprocess
            with st.spinner("Analysing your listing..."):
                processed_data = preprocessing.preprocess_user_input(
                    user_data, 
                    feature_columns, 
                    defaults
                )
            
            # Scale and predict
            processed_data_scaled = scaler.transform(processed_data)
            prediction = model.predict(processed_data_scaled)[0]
            
            # Display results
            st.success("Prediction Complete!")
            st.markdown("---")
            
            # Main prediction
            st.markdown(f"<h2 style='text-align: center;'>Recommended Price: £{prediction:.2f} per night</h2>", unsafe_allow_html=True)
            
            # Metrics row
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Monthly Estimate", f"£{prediction * 25:.2f}", "~25 nights")
            
            with metric_col2:
                per_person = prediction / accommodates
                st.metric("Per Person", f"£{per_person:.2f}")
            
            with metric_col3:
                confidence_interval = prediction * 0.15
                st.metric("Price Range", f"£{prediction - confidence_interval:.0f} - £{prediction + confidence_interval:.0f}")
            
            st.markdown("---")
            
            # Feature impact analysis
            
            st.subheader("What's Driving Your Price?")
            
            impacts = []
            
            # Amenity impacts
            if 'hot tub' in amenities.lower():
                impacts.append(("Hot Tub", "+£22", "premium"))
            if 'pool' in amenities.lower():
                impacts.append(("Pool", "+£18", "premium"))
            if 'gym' in amenities.lower():
                impacts.append(("Gym", "+£8", "positive"))
            if 'parking' in amenities.lower():
                impacts.append(("Free Parking", "+£12", "positive"))
            if 'wifi' in amenities.lower():
                impacts.append(("WiFi", "+£5", "positive"))
            if 'workspace' in amenities.lower() or 'dedicated workspace' in amenities.lower():
                impacts.append(("Workspace", "+£7", "positive"))
            if 'breakfast' in amenities.lower():
                impacts.append(("Breakfast", "+£6", "positive"))
            
            # Location impacts
            if neighbourhood_cleansed == 'City Centre':
                impacts.append(("City Centre Location", "+£18", "premium"))
            elif neighbourhood_cleansed in ['Didsbury West', 'Didsbury East']:
                impacts.append(("Didsbury Location", "+£12", "positive"))
            elif neighbourhood_cleansed in ['Salford District', 'Trafford District']:
                impacts.append(("Good Location", "+£8", "positive"))
            
            # Property size impacts
            if accommodates >= 6:
                impacts.append(("High Capacity (6+ guests)", "+£15", "positive"))
            if bedrooms >= 3:
                impacts.append(("3+ Bedrooms", "+£10", "positive"))
            
            # Review impacts
            if review_scores_rating >= 4.8:
                impacts.append(("Excellent Reviews (4.8+)", "+£8", "positive"))
            elif review_scores_rating < 4.0:
                impacts.append(("Low Reviews (<4.0)", "-£12", "negative"))
            
            if number_of_reviews < 5:
                impacts.append(("Few Reviews (<5)", "-£8", "negative"))
            
            # Superhost
            if host_is_superhost:
                impacts.append(("Superhost Status", "+£7", "positive"))
            
            # Response time
            if host_response_time == 'within an hour':
                impacts.append(("Fast Response Time", "+£4", "positive"))
            elif host_response_time == 'a few days or more':
                impacts.append(("Slow Response Time", "-£6", "negative"))
            
            # Display impacts
            impact_col1, impact_col2 = st.columns(2)
            
            positive_impacts = [i for i in impacts if i[2] in ['positive', 'premium']]
            negative_impacts = [i for i in impacts if i[2] == 'negative']
            
            with impact_col1:
                st.markdown("**Positive Factors**")
                if positive_impacts:
                    for feature, impact, category in positive_impacts:
                        if category == "premium":
                            st.markdown(f"🌟 **{feature}** {impact}")
                        else:
                            st.markdown(f"✓ {feature} {impact}")
                else:
                    st.info("Add premium amenities to increase your price")
            
            with impact_col2:
                st.markdown("**Areas for Improvement**")
                if negative_impacts:
                    for feature, impact, category in negative_impacts:
                        st.markdown(f"⚠️ {feature} {impact}")
                else:
                    st.success("No negative factors detected!")
            
            st.markdown("---")
            
            # Competitive Analysis

            st.subheader("Competitive Positioning")
            
            # Market data by neighbourhood
            competitors = {
                'City Centre': {'low': 45, 'avg': 75, 'high': 120},
                'Salford District': {'low': 35, 'avg': 55, 'high': 85},
                'Trafford District': {'low': 40, 'avg': 65, 'high': 95},
                'Didsbury West': {'low': 50, 'avg': 80, 'high': 110},
                'Didsbury East': {'low': 50, 'avg': 80, 'high': 110},
                'Stockport District': {'low': 30, 'avg': 50, 'high': 75},
                'Bolton District': {'low': 28, 'avg': 45, 'high': 70},
                'Bury District': {'low': 30, 'avg': 48, 'high': 72},
            }
            
            comp_data = competitors.get(neighbourhood_cleansed, {'low': 35, 'avg': 60, 'high': 95})
            
            try:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Market Low', 'Market Average', 'Market High', 'Your Price'],
                    y=[comp_data['low'], comp_data['avg'], comp_data['high'], prediction],
                    marker_color=['lightblue', 'blue', 'darkblue', 'red'],
                    text=[f"£{comp_data['low']}", f"£{comp_data['avg']}", 
                          f"£{comp_data['high']}", f"£{prediction:.0f}"],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title=f"Your Price vs {neighbourhood_cleansed} Market",
                    yaxis_title="Price per Night (£)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                # Fallback if plotly not installed
                st.write(f"**Market Low:** £{comp_data['low']}")
                st.write(f"**Market Average:** £{comp_data['avg']}")
                st.write(f"**Market High:** £{comp_data['high']}")
                st.write(f"**Your Price:** £{prediction:.0f}")
            
            # Market position analysis
            if prediction < comp_data['avg'] * 0.85:
                st.info(f"Your price is **significantly below market average** - excellent for quick bookings and high occupancy")
            elif prediction < comp_data['avg'] * 0.95:
                st.info(f"Your price is **slightly below market average** - good for competitive positioning")
            elif prediction > comp_data['avg'] * 1.15:
                st.warning(f"Your price is **significantly above market average** - premium positioning, may reduce bookings")
            elif prediction > comp_data['avg'] * 1.05:
                st.warning(f"Your price is **slightly above market average** - premium positioning")
            else:
                st.success(f"Your price is **at market average** - balanced competitive positioning")
            
            st.markdown("---")
            
            # Pricing Strategies

            st.subheader("Pricing Strategies")
            
            conservative = prediction * 0.90
            aggressive = prediction * 1.15
            
            strategy_col1, strategy_col2, strategy_col3 = st.columns(3)
            
            with strategy_col1:
                st.markdown("### Conservative")
                st.markdown(f"**£{conservative:.2f}** per night")
                st.write("10% below recommended")
                st.info("**Expected Impact:**\n- Higher occupancy (75-85%)\n- Faster bookings\n- Great for new listings\n- Build reviews quickly")
            
            with strategy_col2:
                st.markdown("### Balanced")
                st.markdown(f"**£{prediction:.2f}** per night")
                st.write("Recommended price")
                st.success("**Expected Impact:**\n- Optimal occupancy (60-70%)\n- Balanced bookings\n- Market-rate pricing\n- Steady revenue")
            
            with strategy_col3:
                st.markdown("### Aggressive")
                st.markdown(f"**£{aggressive:.2f}** per night")
                st.write("15% above recommended")
                st.warning("**Expected Impact:**\n- Lower occupancy (40-55%)\n- Premium positioning\n- Best for peak seasons\n- High-value guests")
            
            st.markdown("---")
            
            # Revenue Projections

            st.subheader("Revenue Projections")
            
            scenarios = {
                'Conservative': {'rate': conservative, 'occupancy': 0.80},
                'Balanced': {'rate': prediction, 'occupancy': 0.65},
                'Aggressive': {'rate': aggressive, 'occupancy': 0.50}
            }
            
            revenue_data = []
            for strategy, data in scenarios.items():
                monthly_nights = 30 * data['occupancy']
                monthly_revenue = data['rate'] * monthly_nights
                annual_revenue = monthly_revenue * 12
                revenue_data.append({
                    'Strategy': strategy,
                    'Nightly Rate': f"£{data['rate']:.0f}",
                    'Est. Occupancy': f"{data['occupancy']*100:.0f}%",
                    'Nights/Month': f"{monthly_nights:.0f}",
                    'Monthly Revenue': f"£{monthly_revenue:.0f}",
                    'Annual Revenue': f"£{annual_revenue:,.0f}"
                })
            
            st.table(pd.DataFrame(revenue_data))
            
            st.markdown("---")
            
            # Recommendations

            st.subheader("Recommendations to Increase Your Price")
            
            recommendations = []
            
            # Check for missing high-value amenities
            if 'wifi' not in amenities.lower():
                recommendations.append(("Add WiFi", "+£5-8/night", "Essential for modern travellers", "high"))
            if 'parking' not in amenities.lower() and neighbourhood_cleansed not in ['City Centre']:
                recommendations.append(("Add Free Parking", "+£10-15/night", "High value in non-central areas", "high"))
            if 'workspace' not in amenities.lower() and 'dedicated workspace' not in amenities.lower():
                recommendations.append(("Add Dedicated Workspace", "+£6-10/night", "Popular with remote workers and business travellers", "medium"))
            if 'hot tub' not in amenities.lower() and accommodates >= 4:
                recommendations.append(("Add Hot Tub", "+£20-25/night", "Premium amenity with high ROI", "premium"))
            if 'pool' not in amenities.lower() and property_type in ['Entire home', 'Entire condo']:
                recommendations.append(("Add Pool", "+£18-22/night", "Highly desirable premium amenity", "premium"))
            
            # Check reviews
            if number_of_reviews < 10:
                recommendations.append(("Get More Reviews", "+£8-12/night", "Aim for 15+ reviews with 4.8+ rating to build trust", "high"))
            elif review_scores_rating < 4.5:
                recommendations.append(("Improve Review Scores", "+£10-18/night", "Focus on cleanliness, communication, and accuracy", "high"))
            
            # Check host status
            if not host_is_superhost and number_of_reviews > 10:
                recommendations.append(("Achieve Superhost Status", "+£7-12/night", "Builds trust and commands premium pricing", "medium"))
            
            # Check response time
            if host_response_time != 'within an hour':
                recommendations.append(("Improve Response Time", "+£4-6/night", "Fast responses increase bookings and satisfaction", "medium"))
            
            # Instant bookable
            if not instant_bookable:
                recommendations.append(("Enable Instant Booking", "+£3-5/night", "Convenience factor for guests, increases visibility", "low"))
            
            # Display recommendations
            if recommendations:
                # Sort by priority
                priority_order = {'high': 0, 'premium': 1, 'medium': 2, 'low': 3}
                recommendations.sort(key=lambda x: priority_order[x[3]])
                
                for i, (action, impact, reason, priority) in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {action} {impact}"):
                        st.write(f"**Why:** {reason}")
                        st.write(f"**Potential Impact:** {impact}")
                        if priority == 'high':
                            st.write("**Priority:** 🔴 High")
                        elif priority == 'premium':
                            st.write("**Priority:** 🌟 Premium Investment")
                        elif priority == 'medium':
                            st.write("**Priority:** 🟡 Medium")
                        else:
                            st.write("**Priority:** 🟢 Low")
            else:
                st.success("Your listing is well-optimised! Focus on maintaining quality and gathering reviews.")
            
            st.markdown("---")
            
            # Strategy recommendation
            st.subheader("Which Strategy Should You Choose?")
            
            if number_of_reviews < 5:
                st.warning("**Recommendation: Start with Conservative pricing**\n\nWith few reviews, competitive pricing will help you attract your first guests and build a strong review foundation.")
            elif number_of_reviews > 50 and review_scores_rating >= 4.8:
                st.success("**Recommendation: Try Aggressive pricing**\n\nYour excellent reviews and track record support premium pricing. Test the higher rate during peak seasons.")
            elif host_is_superhost:
                st.success("**Recommendation: Balanced or Aggressive pricing**\n\nAs a Superhost, you have the credibility to command higher prices. Start with Balanced and test Aggressive during high-demand periods.")
            else:
                st.info("**Recommendation: Balanced pricing**\n\nThis provides the best balance between occupancy and revenue for your listing profile.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    # Information footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Model Information:</strong> XGBoost algorithm trained on real Manchester Airbnb data</p>
    <p>Predictions include text analysis, amenity scoring, and location factors</p>
    <p style='margin-top: 1rem;'><em>By Nicole Reeves</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
