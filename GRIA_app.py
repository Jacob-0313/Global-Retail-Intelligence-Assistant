
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from groq import Groq

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Global Retail Intelligence Assistant",
    page_icon="\U0001F4CA",
    layout="wide"
)

MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    data = pd.read_csv(
        "Global_Superstore2.csv",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )

    data.columns = data.columns.str.strip()

    data["Order Date"] = pd.to_datetime(data["Order Date"])
    data["Ship Date"] = pd.to_datetime(data["Ship Date"])

    data["Year"] = data["Order Date"].dt.year
    data["Month"] = data["Order Date"].dt.month_name()
    data["Month"] = pd.Categorical(data["Month"], categories=MONTH_ORDER, ordered=True)
    data["Quarter"] = data["Order Date"].dt.quarter

    return data


df = load_data()

# ==================================================
# TITLE
# ==================================================

st.title("\U0001F4CA Global Retail Intelligence Assistant (GRIA)")

# ==================================================
# GROQ AI ASSISTANT
# ==================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """
You are GRIA (Global Retail Intelligence Assistant).

You are an expert in:
- Retail Analytics
- Customer Analytics
- Product Analytics
- Supply Chain Analytics
- Business Intelligence

Always provide:
1. Direct Answer
2. Key Findings
3. Business Impact
4. Recommendation

Never mention that you are an AI model.
"""


def ask_gria(question, data):
    if client is None:
        return "Groq API Key not configured."

    dataset_context = f"""
    GLOBAL SUPERSTORE DATASET

    Total Sales: {data['Sales'].sum():,.2f}
    Total Profit: {data['Profit'].sum():,.2f}
    Total Orders: {data['Order ID'].nunique()}
    Total Customers: {data['Customer ID'].nunique()}

    Markets: {', '.join(data['Market'].unique())}
    Categories: {', '.join(data['Category'].unique())}
    Segments: {', '.join(data['Segment'].unique())}

    Highest Revenue Market: {data.groupby('Market')['Sales'].sum().idxmax()}
    Most Profitable Category: {data.groupby('Category')['Profit'].sum().idxmax()}
    Highest Revenue Country: {data.groupby('Country')['Sales'].sum().idxmax()}
    Most Profitable Region: {data.groupby('Region')['Profit'].sum().idxmax()}
    """

    prompt = f"""
    Dataset Context:
    {dataset_context}

    User Question:
    {question}

    Provide:
    1. Direct Answer
    2. Key Findings
    3. Business Impact
    4. Recommendation
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


sample_questions = [
    "Which market generates highest revenue?",
    "Which category is most profitable?",
    "Which products are causing losses?",
    "Who are the top customers?",
    "What is the best shipping mode?",
    "Give me 5 strategic recommendations.",
]

# ==================================================
# SIDEBAR FILTERS
# ==================================================

st.sidebar.header("Filters")

year = st.sidebar.multiselect("Year", sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
market = st.sidebar.multiselect("Market", sorted(df["Market"].unique()), default=sorted(df["Market"].unique()))
region = st.sidebar.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))

filtered_df = df[
    (df["Year"].isin(year))
    & (df["Market"].isin(market))
    & (df["Region"].isin(region))
]

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ==================================================
# KPI SECTION
# ==================================================

sales = filtered_df["Sales"].sum()
profit = filtered_df["Profit"].sum()
orders = filtered_df["Order ID"].nunique()
customers = filtered_df["Customer ID"].nunique()
profit_margin = (profit / sales) * 100 if sales != 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sales", f"${sales:,.0f}")
c2.metric("Profit", f"${profit:,.0f}")
c3.metric("Orders", f"{orders:,}")
c4.metric("Customers", f"{customers:,}")
c5.metric("Margin", f"{profit_margin:.2f}%")

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Sales", "Products", "Customers", "Geography", "Shipping", "Insights"]
)

with tab1:
    st.subheader("Monthly Sales Trend")
    monthly = filtered_df.groupby(["Year", "Month"], observed=True)["Sales"].sum().reset_index()
    fig = px.line(monthly, x="Month", y="Sales", color="Year", markers=True, title="Monthly Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Category Sales")
    cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
    fig = px.bar(cat_sales, x="Category", y="Sales", color="Category", title="Category Sales")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Segment Distribution")
    seg = filtered_df.groupby("Segment")["Sales"].sum().reset_index()
    fig = px.pie(seg, names="Segment", values="Sales", title="Sales by Segment")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Top 10 Countries by Sales")
    country = filtered_df.groupby("Country")["Sales"].sum().nlargest(10).reset_index()
    fig = px.bar(country, x="Country", y="Sales", title="Top 10 Countries by Sales")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Shipping Cost by Market")
    ship = filtered_df.groupby("Market")["Shipping Cost"].sum().reset_index()
    fig = px.bar(ship, x="Market", y="Shipping Cost", title="Shipping Cost by Market")
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Business Insights")
    top_market = filtered_df.groupby("Market")["Sales"].sum().idxmax()
    top_category = filtered_df.groupby("Category")["Profit"].sum().idxmax()
    top_customer = filtered_df.groupby("Customer Name")["Sales"].sum().idxmax()
    top_country = filtered_df.groupby("Country")["Sales"].sum().idxmax()

    st.success(f"Highest Revenue Market: {top_market}")
    st.success(f"Most Profitable Category: {top_category}")
    st.success(f"Top Customer: {top_customer}")
    st.success(f"Highest Revenue Country: {top_country}")

    st.dataframe(filtered_df.head(20))

st.markdown("---")
st.subheader("\U0001F916 Chat With Data")

selected_question = st.selectbox("Choose a sample question", [""] + sample_questions)
user_question = st.text_input("Or type your own question", value=selected_question)

if st.button("Analyze Data"):
    if user_question.strip():
        with st.spinner("Analyzing business data..."):
            answer = ask_gria(user_question, filtered_df)
        st.markdown("### AI Analysis")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
