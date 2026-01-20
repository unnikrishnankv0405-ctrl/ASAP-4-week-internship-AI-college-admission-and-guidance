import streamlit as st
import pandas as pd
from transformers import pipeline
import os

# ---------------- Page config ----------------
st.set_page_config(
    page_title="AI College Selection & Admission Guidance",
    layout="wide"
)

# ---------------- Load data ----------------
df = pd.read_csv("college_data.csv")

# ---------------- Load Hugging Face model ----------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

llm = load_llm()

# ---------------- Title ----------------
st.title("🎓 AI College Selection & Admission Guidance")
st.subheader("Smart guidance for choosing colleges and courses in India")

# ---------------- Sidebar: Student Profile ----------------
st.sidebar.title("🎓 Student Profile")

profession = st.sidebar.selectbox(
    "Select Profession",
    ["Engineering", "Medical", "Arts & Science", "Management"],
)

preferred_course = st.sidebar.text_input(
    "Preferred Course (optional)",
    placeholder="e.g., Computer Science, BBA, MBBS"
)

career_goal = st.sidebar.selectbox(
    "Career Goal",
    [
        "Higher Studies",
        "Job Placement",
        "Government Exams",
        "Entrepreneurship",
        "Research",
        "Still exploring"
    ]
)

location = st.sidebar.text_input(
    "Preferred Location (optional)",
    placeholder="e.g., Chennai, Bangalore, Delhi"
)

help_type = st.sidebar.radio(
    "How can I help you?",
    [
        "College suggestions",
        "College comparison",
        "How to get admission",
        "Career guidance & motivation"
    ]
)

# =========================================================
#                 COLLEGE SUGGESTIONS
# =========================================================
if help_type == "College suggestions":
    st.subheader("🎯 Recommended Colleges for You")

    filtered_df = df[df["Profession"] == profession]

    if preferred_course.strip():
        filtered_df = filtered_df[
            filtered_df["Course"].str.contains(preferred_course, case=False, na=False)
        ]

    if location.strip():
        filtered_df = filtered_df[
            filtered_df["Location"].str.contains(location, case=False, na=False)
        ]

    if filtered_df.empty:
        st.warning("No exact matches found. Try changing filters.")
    else:
        filtered_df = filtered_df.sort_values(by="Rating", ascending=False)

        for _, row in filtered_df.iterrows():
            st.markdown(f"""
            ### 🏫 {row['College']}
            ⭐ **Rating:** {row['Rating']} / 5  
            📍 **Location:** {row['Location']}  
            🎓 **Course:** {row['Course']}

            **Why this college?**
            - {row['Review1']}
            - {row['Review2']}
            """)

        # ---------- AI Guidance ----------
        top_colleges = ", ".join(filtered_df.head(5)["College"].tolist())

        ai_prompt = f"""
Suggest colleges and motivate the student.

Details:
Profession: {profession}
Course: {preferred_course or "Any"}
Career Goal: {career_goal}
Location: {location or "Any"}

Top colleges:
{top_colleges}

Give:
1. Best college explanation
2. Admission steps
3. Motivation
"""

        with st.spinner("🤖 Generating AI guidance..."):
            response = llm(ai_prompt, max_new_tokens=250)

        st.markdown("## 🤖 AI Guidance")
        st.write(response[0]["generated_text"])

# =========================================================
#                 COLLEGE COMPARISON
# =========================================================
if help_type == "College comparison":
    st.subheader("📊 College Comparison")

    colleges = df["College"].unique().tolist()
    col1, col2 = st.columns(2)

    c1 = col1.selectbox("Select College 1", colleges)
    c2 = col2.selectbox("Select College 2", colleges, index=1)

    d1 = df[df["College"] == c1].iloc[0]
    d2 = df[df["College"] == c2].iloc[0]

    col1.markdown(f"""
    ### 🏫 {d1['College']}
    ⭐ {d1['Rating']} / 5  
    📍 {d1['Location']}  
    🎓 {d1['Course']}
    - {d1['Review1']}
    - {d1['Review2']}
    """)

    col2.markdown(f"""
    ### 🏫 {d2['College']}
    ⭐ {d2['Rating']} / 5  
    📍 {d2['Location']}  
    🎓 {d2['Course']}
    - {d2['Review1']}
    - {d2['Review2']}
    """)
# =========================================================
#                 HOW TO GET ADMISSION 
# =========================================================
if help_type == "How to get admission":
    st.subheader("🎓 How to Get Admission")

    target_college = st.text_input(
        "Enter the college you are interested in",
        placeholder="e.g., Hindu College, Delhi"
    )
    target_course = st.text_input(
        "Enter the course you want",
        placeholder="e.g., BA Economics"
    )

    if st.button("Get Admission Guidance"):
        if not target_college or not target_course:
            st.warning("Please enter both college and course.")
        else:
            with st.spinner("🤖 Generating admission guidance..."):

                def generate(prompt, tokens=80):
                    result = llm(
                        prompt,
                        max_new_tokens=tokens,
                        do_sample=True,
                        temperature=0.5
                    )
                    return result[0]["generated_text"].strip()

                # ---------- Eligibility ----------
                eligibility = generate(
                    f"""
List 2–3 short eligibility points for admission to
{target_course} at {target_college} in India.
Use simple bullet points.
"""
                )

                # ---------- Entrance Exams ----------
                exams = generate(
                    f"""
Mention entrance exams (if any) required for
{target_course} at {target_college}.
If admission is merit-based, clearly say so.
Keep it short.
"""
                )

                # ---------- Admission Process ----------
                process = generate(
                    f"""
Explain the admission process for {target_course}
at {target_college} in exactly 4 simple numbered steps.
"""
                , tokens=120)

                # ---------- Preparation Tips ----------
                tips = generate(
                    f"""
Give 4 practical preparation tips for a student
applying to {target_course} in India.
Use bullet points.
"""
                , tokens=120)

                # ---------- Motivation ----------
                motivation = generate(
                    """
Write a short motivational message (4–5 lines)
to encourage a student applying for college.
Keep it positive and simple.
"""
                , tokens=120)

            # ---------- Display ----------
            st.success("Admission Guidance")

            st.markdown("### ✅ Eligibility")
            st.markdown(eligibility)

            st.markdown("### 📝 Entrance Exams")
            st.markdown(exams)

            st.markdown("### 🧭 Admission Process")
            st.markdown(process)

            st.markdown("### 📚 Preparation Tips")
            st.markdown(tips)

            st.markdown("### 🌟 Motivation")
            st.markdown(motivation)

# =========================================================
#           CAREER GUIDANCE & MOTIVATION 
# =========================================================
if help_type == "Career guidance & motivation":
    st.subheader("🌱 Career Guidance & Motivation")

    course_for_career = st.sidebar.text_input(
        "Confirm your course",
        value=preferred_course or "",
        key="career_course"
    )

    if st.button("Get Career Guidance"):
        if not course_for_career:
            st.warning("Please enter a course to get career guidance.")
        else:
            with st.spinner("🤖 Generating career guidance..."):

                def generate(text):
                    out = llm(
                        text,
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.5
                    )
                    return out[0]["generated_text"].strip()

                careers = generate(
                    f"List 4 career options after studying {course_for_career} in India as short bullet points."
                )

                higher_studies = generate(
                    f"List common higher study options in India after {course_for_career}."
                )

                skills = generate(
                    f"List important skills students should build for success in {course_for_career}."
                )

                motivation = generate(
                    "Write a short, positive motivational message for a college student."
                )

            st.success("Career Guidance Ready")

            st.markdown("### 🎯 Career Options")
            st.write(careers)

            st.markdown("### 🎓 Higher Studies")
            st.write(higher_studies)

            st.markdown("### 🛠 Skills to Build")
            st.write(skills)

            st.markdown("### 🌟 Motivation")
            st.write(motivation)
