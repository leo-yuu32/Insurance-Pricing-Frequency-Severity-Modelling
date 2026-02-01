"""
Streamlit Dashboard for Insurance Pricing Strategy Lab

Interactive dashboard to demonstrate the impact of actuarial loadings in real-time.
"""

import streamlit as st
import pandas as pd

# Import pricing engine functions
import pricing_engine

# Page Configuration
st.set_page_config(
    page_title="Pricing Strategy Lab", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Custom CSS for premium styling
st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap'
    );
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #1e293b;
    }
    
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
    }

    /* Navbar-like Header */
    .nav-container {
        background-color: #ffffff;
        padding: 1rem 2rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0px;
        letter-spacing: -0.02em;
        text-shadow: 2px 2px 0px rgba(0,0,0,0.05);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 400;
        margin-top: 0.5rem;
        max-width: 600px;
    }
    
    /* Metric Cards - Dark & White */
    div[data-testid="stMetric"] {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stMetric"] > div {
        color: #ffffff !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        color: #94a3b8; /* Light gray for label */
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* REMOVED: div[data-testid="stMetricDelta"] override to restore default GREEN color */

    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        color: #64748b;
        font-weight: 600;
        font-size: 1rem;
        padding-bottom: 12px;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #35404f !important; /* Force Black text for selected */
        border-bottom: 3px solid #0f172a;
        background-color: transparent; /* Prevent background issues */
    }

    /* Slider Customization - LIGHT BLUE */
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
        background-color: transparent !important;
        box-shadow: transparent !important;
    }

    div.stSlider > div[data-baseweb="slider"] > div > div {
        background-color: transparent !important;
    }

    div.stSlider > div[data-baseweb="slider"] > div > div[style*="width"] {
        background-color: transparent !important;
    }
    
    /* REMOVE SLIDER HOVER TOOLTIP */
    div[data-testid="stSlider"] div[role="tooltip"] {
        display: none !important;
    }

    /* HIDE SIDEBAR COLLAPSE BUTTON ("Make smaller window button too") */
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    
    </style>
""",
    unsafe_allow_html=True,
)

# Title
# Title with modern styling
# Navbar / Header Section
with st.container():
    st.markdown(
        '<h1 class="main-header">Insurance Pricing Analysis Dashboard</h1>', unsafe_allow_html=True
    )
    st.markdown("---")  # LINE ADDED HERE (Between Title and Description)
    st.markdown(
        '<p class="sub-header">A real-time simulation engine for Motor Third-Party '
        "Liability pricing using the French Motor Third-Party Liability Dataset. "
        "This dashboard visualises the gap between Technical Raw Model Predictions and "
        "Commercial Pricing due to actuarial loadings. Allowing us to visualise the impact of our"
        "loadings on the model's performance.</p>",
        unsafe_allow_html=True,
    )
    repo_url = (
        "https://github.com/idk-what-to-put-here-32/"
        "Insurance-Pricing-Frequency-Severity-Modelling"
    )
    st.link_button("View Source Code on GitHub", repo_url)


# ============================================================================
# CACHED FUNCTIONS - Heavy lifting happens here
# ============================================================================


@st.cache_resource
def load_and_train_models():
    """
    Load data and train GLM models.
    This is cached so models are only trained once.
    """
    with st.spinner("Loading data and training models (this may take a minute)..."):
        result = pricing_engine.run_pricing_pipeline(custom_loadings=None, return_train_data=True)
    return result


# ============================================================================
# DYNAMIC RECALCULATION FUNCTION
# ============================================================================


def recalculate_with_loadings(base_result, custom_loadings):
    """
    Takes cached GLM predictions and applies custom loadings to calculate final price.

    Parameters:
    -----------
    base_result : dict
        Result from run_pricing_pipeline with return_train_data=True
    custom_loadings : dict
        Custom loadings dictionary

    Returns:
    --------
    dict with updated test_results and metrics
    """
    test_results = base_result["test_results"].copy()

    # Apply custom loadings
    test_results["Final_Price"] = test_results.apply(
        lambda row: pricing_engine.apply_loadings(row, custom_loadings), axis=1
    )

    # Recalculate metrics
    gini_commercial = pricing_engine.calculate_gini(
        test_results, actual_col="TotalLoss", pred_col="Final_Price", exposure_col="Exposure"
    )

    avg_premium = (test_results["Final_Price"] * test_results["Exposure"]).sum() / test_results[
        "Exposure"
    ].sum()

    return {
        "test_results": test_results,
        "gini_commercial": gini_commercial,
        "avg_premium": avg_premium,
    }


# ============================================================================
# SIDEBAR - Strategy Controls
# ============================================================================

st.sidebar.header("Commercial Levers")

st.sidebar.markdown("---")
st.sidebar.subheader("Driver Age Adjustments")

young_driver_loading = st.sidebar.slider(
    "Young Driver Multiplier (Age 18-21)",
    min_value=1.0,
    max_value=3.0,
    value=2.4,
    step=0.1,
    help="Multiplier for drivers aged 18-21",
)

young_adult_loading = st.sidebar.slider(
    "Young Adult Loading (22-25)",
    min_value=1.0,
    max_value=2.0,
    value=1.4,
    step=0.1,
    help="Multiplier for drivers aged 22-25",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Vehicle Age Adjustments")

new_car_discount = st.sidebar.slider(
    "New Car Multiplier (Age 0-1)",
    min_value=0.5,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Multiplier for new vehicles (0-1 years)",
)

old_car_discount = st.sidebar.slider(
    "Old Car Discount (20+ yrs)",
    min_value=0.3,
    max_value=1.0,
    value=0.55,
    step=0.05,
    help="Discount multiplier for very old vehicles (20+ years)",
)

mid_age_surcharge = st.sidebar.slider(
    "Mid-Age Surcharge (5-10 yrs)",
    min_value=1.0,
    max_value=1.5,
    value=1.12,
    step=0.01,
    help="Surcharge multiplier for mid-age vehicles (5-10 years)",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Vehicle Power Adjustments")

power_9_surcharge = st.sidebar.slider(
    "Power 9 Surcharge",
    min_value=1.0,
    max_value=1.5,
    value=1.30,
    step=0.05,
    help="Surcharge for vehicle power level 9",
)

power_4_discount = st.sidebar.slider(
    "Power 4 Discount",
    min_value=0.7,
    max_value=1.0,
    value=0.90,
    step=0.05,
    help="Discount for vehicle power level 4",
)

# Build custom loadings dictionary
custom_loadings = {
    "DriverAge_Bin": {
        "18-21": young_driver_loading,
        "22-25": young_adult_loading,
    },
    "VehAge_Bin": {
        "New (0-1)": new_car_discount,
        "5-10": mid_age_surcharge,
        "20+": old_car_discount,
    },
    "VehPower_Bin": {
        "9": power_9_surcharge,
        "4": power_4_discount,
    },
}

# ============================================================================
# MAIN AREA - Load cached data
# ============================================================================

# Load cached models and base results
base_result = load_and_train_models()

# Recalculate with custom loadings
updated_result = recalculate_with_loadings(base_result, custom_loadings)

test_results = updated_result["test_results"]
gini_technical = base_result["gini_technical"]
gini_commercial = updated_result["gini_commercial"]
avg_premium = updated_result["avg_premium"]

# ============================================================================
# KEY METRICS DISPLAY
# ============================================================================

# REMOVED st.markdown("---") Line Here as requested (one of the two lines)
st.subheader("Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Technical Gini",
        value=f"{gini_technical:.4f}",
        help="Gini coefficient for PurePremium (model output)",
    )

with col2:
    gini_delta = gini_commercial - gini_technical
    st.metric(
        label="Commercial Gini",
        value=f"{gini_commercial:.4f}",
        delta=f"{gini_delta:+.4f}",
        help="Gini coefficient for Final Price (after loadings)",
    )

with col3:
    st.metric(
        label="Average Premium",
        value=f"€{avg_premium:.2f}",
        help="Exposure-weighted average premium",
    )

with col4:
    total_premium = (test_results["Final_Price"] * test_results["Exposure"]).sum()
    st.metric(
        label="Total Premium",
        value=f"€{total_premium:,.0f}",
        help="Total premium across all policies",
    )

# ============================================================================
# TABS FOR ANALYSIS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["Model Performance", "Detailed Analysis", "Dislocation Analysis"])

with tab1:
    st.subheader("Lift Charts: Technical vs Commercial")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Technical View (PurePremium)**")

        # Use matplotlib lift chart function
        fig_tech = pricing_engine.plot_lift_curve(
            df=test_results,
            pred_col="PurePremium_OB",
            actual_col="TotalLoss",
            title="Lift Chart: Technical View (Pure Premium vs Actual)",
            view_type="Technical",
            return_fig=True,
        )
        st.pyplot(fig_tech, use_container_width=True)

    with col2:
        st.markdown("**Commercial View (Final Price)**")

        # Use matplotlib lift chart function
        fig_com = pricing_engine.plot_lift_curve(
            df=test_results,
            pred_col="Final_Price",
            actual_col="TotalLoss",
            title="Lift Chart: Commercial View (Final Price vs Actual)",
            view_type="Commercial",
            return_fig=True,
        )
        st.pyplot(fig_com, use_container_width=True)

    # Add explanatory text below the charts
    st.markdown(
        """
        ---
        **How to read these charts:**
        The Lift Chart segments the portfolio into 10 risk buckets (Deciles) from lowest to
        highest predicted risk.

        * **The Technical View (The Left Graph)** shows the base model's predicted rates for each
        risk bucket in green vs the actual rates in blue.
        * **The Commercial View (The Right Graph)** shows the final prices (after applying our
        loadings) for each risk bucket in green vs the actual rates in blue.
        * **Steep Slope:** A steep upward slope confirms the model successfully differentiates
        between safe (Decile 1) and risky (Decile 10) drivers.
        * **Close Alignment:** If the **Green line** (Predicted) closely tracks the **Blue line**
        (Actual), the model is accurately calibrated.
        * **Impact on Gini:** The Gini Coefficient is a measure of the model's ability to predict
        risk, the higher the Gini, the better the model is at predicting risk. Notice how the
        Commercial Gini is higher than the Technical Gini, this is because we are applying loadings
        to the model to make it better.
        """
    )

    # Add Lorenz Curve centered below the lift charts
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:
        st.markdown("**Lorenz Curve: Risk Differentiation**")
        lorenz_fig = pricing_engine.plot_lorenz_curve(
            df=test_results, pred_col="Final_Price", return_fig=True
        )
        st.pyplot(lorenz_fig, use_container_width=True)
        st.info(
            "This chart measures the model's "
            "ability to segment risk. The Gini Coefficient scores this segmentation "
            "(0 = Random, 1 = Perfect). A higher Gini means the model is better at "
            "identifying high-risk drivers."
            ""
            "The x-axis shows the proportion of the drivers by risk, "
            "the left hand side the least risky drivers and the right hand side the most risky"
            "drivers. The y-axis shows the proportion of the total losses that actually occurred."
            "If the model is perfect, the curve will be a straight line at 45 degrees (perfect"
            "equality). The closer the blue curve is to the diagonal line, the better the model"
            "is at predicting risk. The Gini Coefficient is the area between the blue curve and the"
            "diagonal line. The higher the Gini, the better the model is at predicting risk."
        )

with tab2:
    st.subheader("Detailed Analysis")

    # Analysis selection (Lorenz Curve moved to Tab 1)
    selected_analysis = st.selectbox(
        "Select Analysis:", options=["Driver Age", "Vehicle Age", "Vehicle Power"], index=0
    )

    # Generate one-way analysis for selected feature
    feature_agg = pricing_engine.plot_one_way(
        df=test_results,
        feature=selected_analysis,
        filename=None,  # Don't save
        return_fig=False,  # Return aggregated data
    )

    # Get the figure for display
    oneway_fig = pricing_engine.plot_one_way(
        df=test_results, feature=selected_analysis, return_fig=True
    )

    # Display the plot
    st.pyplot(oneway_fig, use_container_width=True)

    # Add explanatory text for non-technical users
    st.info(
        "These charts compare the Actual Risk (Red), "
        "the Technical Model (Blue), and the Final Commercial Price (Green). "
        "Gaps between Blue and Green indicate where Actuarial Strategy "
        "(Loadings/Discounts) has been applied."
        ""
        "The x-axis shows the risk buckets (Deciles) from lowest to highest predicted risk."
        "The blue line shows the model's predicted rates for each risk bucket."
        "The green line shows the final prices (after applying our loadings) for each risk bucket."
        "The red line shows the actual rates for each risk bucket."
        "Notice how the green line (after loadings) matches the red line (actual) better than the"
        "blue line (base model)."
    )

    # Display detailed table for selected feature's key bin (if Driver Age selected)
    if selected_analysis == "Driver Age":
        st.markdown("---")
        st.subheader("Verification: Driver Age 18-21 Bin")

        driver_18_21 = feature_agg[feature_agg["DriverAge_Bin"] == "18-21"].iloc[0]

        verification_data = {
            "Metric": [
                "Actual Average Loss",
                "Model Predicted Rate",
                "Final Price Rate",
                "Risk Gap (Actual - Model)",
                "Loading Impact (Final - Model)",
                "Loading Multiplier",
                "Exposure Volume",
            ],
            "Value": [
                f"€{driver_18_21['Actual_Rate']:.2f}",
                f"€{driver_18_21['Model_Rate']:.2f}",
                f"€{driver_18_21['Final_Rate']:.2f}",
                f"€{driver_18_21['Actual_Rate'] - driver_18_21['Model_Rate']:.2f}",
                f"€{driver_18_21['Final_Rate'] - driver_18_21['Model_Rate']:.2f}",
                f"{driver_18_21['Final_Rate'] / driver_18_21['Model_Rate']:.2f}x",
                f"{driver_18_21['Exposure']:,.2f}",
            ],
        }

        verification_df = pd.DataFrame(verification_data)
        st.dataframe(verification_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Dislocation Analysis")
    st.markdown("**Distribution of price changes between Technical and Commercial pricing**")

    # Generate dislocation histogram
    dislocation_fig = pricing_engine.plot_dislocation_histogram(df=test_results, return_fig=True)

    st.pyplot(dislocation_fig, use_container_width=True)

    # Add explanatory text for non-technical users
    st.info(
        "This chart is a dislocation histogram, it shows the distribution of price changes between"
        "the Technical and Commercial pricing."
        "Positive values indicate customers paying more than the technical premium (the base model)"
        ".Negative values indicate customers paying less (Discounts) due to our loadings."
        "Since most of our graph is green, this means we are applying loadings to the model to also"
        "make it better for the commercial market."
    )

    # Calculate metrics
    test_results["PctChange"] = (
        (test_results["Final_Price"] - test_results["PurePremium_OB"])
        / test_results["PurePremium_OB"]
        * 100
    )

    max_increase = test_results["PctChange"].max()
    max_decrease = test_results["PctChange"].min()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Max Price Increase",
            value=f"{max_increase:.2f}%",
            help="Maximum percentage increase in price from Technical to Commercial",
        )

    with col2:
        st.metric(
            label="Max Price Decrease",
            value=f"{max_decrease:.2f}%",
            help="Maximum percentage decrease in price from Technical to Commercial",
        )

# Footer
st.markdown("---")
st.markdown(
    "**Tip:** Adjust the sliders in the sidebar to see real-time impact on "
    "pricing and Gini coefficients!"
)
