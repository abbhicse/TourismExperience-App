import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Tourist Recommender", layout="wide")

# -------------------------------
# Load Data & Models
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("output/feature_engineered_data.xlsx")
    df = df.dropna(subset=["VisitMode"])
    return df


@st.cache_resource
def load_models():
    classifier = joblib.load("models/visitmode_classifier_randomforest.pkl")
    regressor = joblib.load("models/rating_regressor.pkl")
    label_encoder = joblib.load("models/visitmode_label_encoder.pkl")

    classifier_features = joblib.load("models/classifier_features.pkl")
    regressor_features = joblib.load("models/regressor_features.pkl")

    template_classifier = joblib.load("models/input_template_row_classifier.pkl")
    template_regressor = joblib.load("models/input_template_row_regressor.pkl")

    encoders = joblib.load("models/regressor_label_encoders.pkl")  # Country, Region, Continent

    # UPDATED: load the smaller sparse collaborative model (joblib)
    collab_model = joblib.load("models/collaborative_model.pkl")

    # Content model unchanged
    content_model = joblib.load("models/content_based_model.pkl")

    return (
        classifier,
        regressor,
        label_encoder,
        classifier_features,
        regressor_features,
        template_classifier,
        template_regressor,
        encoders,
        collab_model,
        content_model,
    )


df = load_data()
(
    classifier,
    regressor,
    label_encoder,
    classifier_features,
    regressor_features,
    template_classifier,
    template_regressor,
    encoders,
    collab_model,
    content_model,
) = load_models()

# Fast lookup for attraction names
@st.cache_data
def build_attr_name_map(df_):
    m = (
        df_[["AttractionId", "Attraction"]]
        .dropna()
        .drop_duplicates(subset=["AttractionId"])
        .set_index("AttractionId")["Attraction"]
        .to_dict()
    )
    return m

ATTR_NAME = build_attr_name_map(df)

# -------------------------------
# Streamlit Layout
# -------------------------------
st.title("Smart Tourist Recommender App")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Classification", "Rating Prediction", "Visual Insights", "Recommendations"]
)

# -----------------------------------
# Tab 1: Classification (Visit Mode)
# -----------------------------------
with tab1:
    st.header("Predict Visit Mode")

    # Step 1: Select travel location
    user_country = st.selectbox("Country", df["Country"].dropna().unique())
    user_region = st.selectbox(
        "Region", df[df["Country"] == user_country]["Region"].dropna().unique()
    )
    user_continent = st.selectbox(
        "Continent", df[df["Region"] == user_region]["Continent"].dropna().unique()
    )

    # Step 2: Encode country/region/continent using label encoders
    encoded_inputs = {}
    for col, val in zip(
        ["Country", "Region", "Continent"], [user_country, user_region, user_continent]
    ):
        encoder = encoders[col]
        if val not in encoder.classes_:
            st.error(f"{col} value '{val}' not recognized by encoder.")
            st.stop()
        encoded_inputs[f"{col}_Encoded"] = int(encoder.transform([val])[0])

    # Step 3: Dynamic form for all classifier features
    st.subheader("Customize Features")

    input_row = {}

    with st.expander("Location & Encoded Features"):
        for feature in classifier_features:
            if feature in encoded_inputs:
                input_row[feature] = encoded_inputs[feature]
            elif any(key in feature.lower() for key in ["country", "region", "continent", "encoded"]):
                default_val = template_classifier.get(feature, 0)
                input_row[feature] = st.number_input(
                    f"{feature}",
                    value=int(default_val),
                    step=1,
                    help="Categorical feature encoded as a number",
                )

    with st.expander("Temporal Features (Month/Weekday)"):
        for feature in classifier_features:
            if "month" in feature.lower() or "weekday" in feature.lower():
                default_val = template_classifier.get(feature, 6)
                # (kept as your original; if weekday is 1-7, you can change range here)
                input_row[feature] = st.slider(
                    f"{feature}",
                    1,
                    12,
                    int(default_val),
                    help="Represents time-based travel preferences",
                )

    with st.expander("Numeric & Rating Features"):
        for feature in classifier_features:
            if feature in input_row:
                continue

            default_val = int(template_classifier.get(feature, 0))

            if feature.lower().startswith("price") or "rating" in feature.lower():
                input_row[feature] = st.slider(
                    f"{feature}",
                    0,
                    5,
                    default_val,
                    1,
                    help="Scale of 0-5 for preferences or ratings (integer only)",
                )
            else:
                input_row[feature] = st.number_input(
                    f"{feature}",
                    value=default_val,
                    step=1,
                    help="Numeric feature (integer only)",
                )

    # Step 4: Build input DataFrame
    input_df = pd.DataFrame([input_row])

    # Step 5: Predict Visit Mode
    if st.button("Predict Visit Mode"):
        st.subheader("Visit Mode Prediction")

        missing_clf = [f for f in classifier_features if f not in input_df.columns]
        if missing_clf:
            st.error(f"Missing classifier features: {missing_clf}")
        else:
            pred_input_clf = input_df[classifier_features]
            pred_encoded = int(classifier.predict(pred_input_clf)[0])
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]
            st.success(f"Predicted Visit Mode: **{pred_label}**")

            if hasattr(classifier, "predict_proba"):
                proba = classifier.predict_proba(pred_input_clf)[0]
                st.subheader("Prediction Confidence:")
                for label, p in zip(label_encoder.classes_, proba):
                    st.write(f"{label}: {p:.2%}")

            # Save for Tab 2
            st.session_state.visit_mode_encoded = pred_encoded
            st.session_state.classifier_input = input_row

# -----------------------------------
# Tab 2: Regression (Rating Score)
# -----------------------------------
with tab2:
    st.header("Predict Visit Rating Score")

    if "visit_mode_encoded" not in st.session_state or "classifier_input" not in st.session_state:
        st.warning("Please complete the Visit Mode prediction in Tab 1 first.")
    else:
        reg_input = dict(st.session_state.classifier_input)
        reg_input["VisitMode_Encoded"] = st.session_state.visit_mode_encoded

        st.subheader("Additional Inputs for Rating Prediction")

        with st.expander("User-Level Features"):
            for feature in regressor_features:
                if feature in reg_input:
                    continue

                default_val = template_regressor.get(feature, 0)

                if feature == "UserAvgRating":
                    reg_input[feature] = st.slider(
                        "User Avg Rating", 1.0, 5.0, float(default_val),
                        help="Average of all user ratings"
                    )
                elif feature == "UserVisitCount":
                    reg_input[feature] = st.slider(
                        "User Visit Count", 0, 100, int(default_val),
                        help="Number of previous visits"
                    )
                elif feature == "UserVisitCount_Scaled":
                    uvc = reg_input.get("UserVisitCount", 10)
                    reg_input[feature] = float(uvc) / 100.0
                    st.text(f"UserVisitCount_Scaled auto-calculated as {reg_input[feature]:.2f}")
                else:
                    reg_input[feature] = st.number_input(
                        f"{feature}",
                        value=float(default_val),
                        help="Additional numeric feature used for rating prediction",
                    )

        reg_input_df = pd.DataFrame([reg_input])

        with st.expander("Preview Input to Rating Model"):
            st.dataframe(reg_input_df[regressor_features])

        if st.button("Predict Visit Rating"):
            missing = [f for f in regressor_features if f not in reg_input_df.columns]
            if missing:
                st.error(f"Missing required features: {missing}")
            else:
                pred_rating = float(regressor.predict(reg_input_df[regressor_features])[0])
                st.success(f"Predicted Visit Rating: **{pred_rating:.2f}**")

# -------------------------------
# Tab 3: Visual Insights
# -------------------------------
with tab3:
    st.header("Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Attractions by Avg Rating")
        top_attr = (
            df.groupby("Attraction")["Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig1, ax1 = plt.subplots()
        sns.barplot(data=top_attr, y="Attraction", x="Rating", ax=ax1, palette="viridis")
        ax1.set_title("Top Attractions")
        st.pyplot(fig1)

    with col2:
        st.subheader("Most Visited Regions")
        top_regions = (
            df.groupby("Region")["TransactionId"]
            .count()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig2, ax2 = plt.subplots()
        sns.barplot(data=top_regions, x="TransactionId", y="Region", ax=ax2, palette="rocket")
        ax2.set_title("Top Regions")
        st.pyplot(fig2)

# -------------------------------
# Tab 4: Recommendations
# -------------------------------
with tab4:
    st.header("Personalized Recommendations")

    user_id_input = st.number_input("Enter User ID:", min_value=1, value=32)
    rec_type = st.radio("Choose Recommendation Type:", ["Collaborative Filtering", "Content-Based Filtering"])

    def get_attr_name(attr_id):
        return ATTR_NAME.get(attr_id, "Unknown")

    def fallback_popular(user_id, top_n=5):
        already_rated = set(df.loc[df["UserId"] == user_id, "AttractionId"].tolist())

        popular_items = collab_model.get("popular_items_sorted", None)
        if popular_items is None:
            popular_items = (
                df.groupby("AttractionId")["Rating"]
                .mean()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

        recs = [iid for iid in popular_items if iid not in already_rated]
        return recs[:top_n]

    def recommend_collab(user_id, top_n=5, n_neighbors=10):
        """
        Sparse user-based CF:
        - uses X (CSR) and X_norm (row-normalized CSR) from collab_model
        - computes cosine similarity ONLY for requested user
        - scores items via weighted mean of neighbor ratings
        """
        X = collab_model["X"]
        X_norm = collab_model["X_norm"]
        user_to_row = collab_model["user_to_row"]
        item_ids = np.array(collab_model["item_ids"])  # col -> AttractionId

        n_users, n_items = X.shape

        if user_id not in user_to_row:
            return fallback_popular(user_id, top_n=top_n)

        u = user_to_row[user_id]

        # Similarity to all users (no full n_users x n_users matrix stored)
        sims = (X_norm @ X_norm[u].T).toarray().ravel().astype(np.float32)
        sims[u] = -1.0  # exclude self

        k = min(int(n_neighbors), n_users - 1)
        if k <= 0:
            return fallback_popular(user_id, top_n=top_n)

        # argpartition kth must be in [0, n-1]
        nn_idx = np.argpartition(-sims, kth=k - 1)[:k]
        nn_idx = nn_idx[np.argsort(-sims[nn_idx])]
        nn_sims = sims[nn_idx]

        # If no positive similarities, fallback
        if np.all(nn_sims <= 0):
            return fallback_popular(user_id, top_n=top_n)

        # Neighbor ratings matrix: (k x n_items)
        Rn = X[nn_idx]

        # Weighted sum of ratings
        weighted = Rn.multiply(nn_sims.reshape(-1, 1))
        numerator = np.array(weighted.sum(axis=0)).ravel()

        # Denominator: sum of similarities for neighbors who rated each item
        mask = Rn.copy()
        mask.data = np.ones_like(mask.data, dtype=np.float32)
        denominator = np.array(mask.multiply(nn_sims.reshape(-1, 1)).sum(axis=0)).ravel()

        scores = np.full(n_items, -np.inf, dtype=np.float32)
        valid = denominator > 0
        scores[valid] = numerator[valid] / denominator[valid]

        # Exclude items already rated by this user (in training matrix)
        scores[X[u].indices] = -np.inf

        # Choose top-N
        top_n = min(int(top_n), n_items)
        if top_n <= 0:
            return fallback_popular(user_id, top_n=5)

        top_cols = np.argpartition(-scores, kth=top_n - 1)[:top_n]
        top_cols = top_cols[np.argsort(-scores[top_cols])]

        recs = item_ids[top_cols].tolist()
        return recs if recs else fallback_popular(user_id, top_n=top_n)

    if st.button("Recommend Attractions"):
        if rec_type == "Collaborative Filtering":
            recs = recommend_collab(user_id_input, top_n=5, n_neighbors=10)

        else:
            # Content-Based (your original logic)
            attr_df = content_model["attraction_df"]
            sim_matrix = content_model["cosine_sim"]
            indices = pd.Series(attr_df.index, index=attr_df["AttractionId"])

            user_ratings = df[(df["UserId"] == user_id_input) & (df["Rating"] >= 4)]
            if user_ratings.empty:
                st.warning("No high ratings. Showing fallback.")
                recs = fallback_popular(user_id_input, top_n=5)
            else:
                top_rated = user_ratings.sort_values(by="Rating", ascending=False).iloc[0]
                attr_id = top_rated["AttractionId"]

                if attr_id not in indices:
                    st.warning("No metadata for that attraction. Fallback used.")
                    recs = fallback_popular(user_id_input, top_n=5)
                else:
                    idx = int(indices[attr_id])
                    sim_scores = list(enumerate(sim_matrix[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
                    rec_indices = [i for i, _ in sim_scores]
                    recs = attr_df.iloc[rec_indices]["AttractionId"].tolist()

        st.subheader("Recommendations")
        if not recs:
            st.info("No recommendations available.")
        else:
            for rid in recs:
                st.markdown(f"**{get_attr_name(rid)}**  \nAttractionId: `{rid}`")
