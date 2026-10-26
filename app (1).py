import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote_plus

st.set_page_config(page_title="Personalized Learning Course Recommendation System", page_icon="📚", layout="wide")
st.title("📚 Personalized Learning Course Recommendation System")
st.write("Discover courses, open YouTube lessons, read notes, and test your knowledge.")

CSV_FILE = "optimized_learning_resource_recommendations_200.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(CSV_FILE)
    required = ["course_id", "topic", "difficulty", "content_type"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))
    topic_names = {"AI": "Artificial Intelligence", "Art": "Art and Creativity",
                   "Data Science": "Data Science", "History": "World History",
                   "Math": "Mathematics", "Music": "Music"}
    levels = {"Beginner": "Introduction to", "Intermediate": "Intermediate", "Advanced": "Advanced"}
    formats = {"Video": "Video Lesson", "Quiz": "Practice Quiz", "Text": "Learning Notes"}
    def make_title(row):
        topic = topic_names.get(str(row["topic"]), str(row["topic"]))
        level = levels.get(str(row["difficulty"]), str(row["difficulty"]))
        fmt = formats.get(str(row["content_type"]), str(row["content_type"]))
        return f"{level} {topic} - {fmt}"
    data["title"] = data.apply(make_title, axis=1)
    data = data.drop_duplicates(subset="course_id").reset_index(drop=True)
    for col in ["topic", "difficulty", "content_type"]:
        data[col] = data[col].astype(str)
    return data

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Cannot find {CSV_FILE}. Save it in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the CSV: {e}")
    st.stop()

LESSONS = {
"AI": ("Introduction to Artificial Intelligence", """
Artificial Intelligence (AI) builds computer systems that perform tasks associated with human intelligence.

**Key ideas**
- **Data:** information or examples used by a system.
- **Model:** a program or learned representation used to produce outputs.
- **Training:** learning patterns from examples.
- **Prediction:** an output for new input.

**Example:** An email spam filter can learn patterns in messages and classify new emails.
**Remember:** AI is the broad field; machine learning is one approach within AI.
"""),
"Data Science": ("Introduction to Data Science", """
Data Science uses data, programming, statistics, and domain knowledge to answer questions.

**A simple workflow**
1. Collect data.
2. Clean missing, incorrect, or duplicate values.
3. Explore data using tables and charts.
4. Build a model if prediction is needed.
5. Explain results and limitations.

**Example:** A shop can study sales data to find best-selling products.
**Remember:** Good results depend on suitable data and careful interpretation.
"""),
"Math": ("Mathematics Learning Notes", """
Mathematics helps describe quantities, patterns, and relationships.

- **Variable:** a symbol representing a value.
- **Equation:** a statement that two expressions are equal.
- **Mean:** sum of values divided by the number of values.
- **Probability:** a measure from 0 to 1 of how likely an event is.

**Example:** The mean of 4, 6, and 8 is (4 + 6 + 8) / 3 = 6.
"""),
"Art": ("Introduction to Art and Creativity", """
Art communicates ideas, feelings, and stories through creative work.

- **Line:** can guide attention or create movement.
- **Shape:** a two-dimensional enclosed area.
- **Colour:** can create contrast, mood, and emphasis.
- **Texture:** actual or suggested surface quality.
- **Composition:** how elements are arranged.

**Example:** A poster can use contrast and large text to make its main message noticeable.
"""),
"History": ("Introduction to World History", """
History studies the past using evidence such as documents, objects, buildings, images, and oral accounts.

**How historians work**
1. Ask a clear question.
2. Find and compare sources.
3. Check who created each source and why.
4. Consider its time and context.
5. Support conclusions with evidence.

Sources may contain bias or missing information, so comparing sources is useful.
"""),
"Music": ("Introduction to Music", """
Music organizes sound and silence to express ideas, emotions, and culture.

- **Rhythm:** pattern of sounds and beats over time.
- **Melody:** a sequence of notes heard as a musical line.
- **Tempo:** how fast or slow music moves.
- **Dynamics:** how loud or soft music is.
- **Timbre:** sound quality that helps distinguish instruments or voices.

Listening for one element at a time can help you describe music clearly.
""")
}

QUIZZES = {
"AI": [
("Which statement best describes Artificial Intelligence?", ["A field that builds systems for tasks associated with intelligence", "A device used only to store files", "A programming language", "A type of computer screen"], 0, "AI builds systems that perform tasks such as recognizing patterns or making predictions."),
("In machine learning, what is a model?", ["A learned function or representation used to make predictions", "A folder containing pictures only", "A computer power cable", "A list of website links"], 0, "A trained model uses patterns learned from data to produce outputs for new inputs."),
("Which is an example of an AI application?", ["An email spam detector", "A blank notebook", "A disconnected keyboard", "A charging cable"], 0, "Spam detectors can classify emails using patterns in messages.")
],
"Data Science": [
("What should usually happen before analysing a dataset?", ["Check and clean the data", "Delete every column", "Assume all values are correct", "Replace values with random numbers"], 0, "Cleaning helps identify missing, inconsistent, incorrect, or duplicate data."),
("Which Python library is commonly used for tabular data?", ["Pandas", "Turtle only", "Pygame only", "Tkinter only"], 0, "Pandas provides DataFrame and Series structures for tabular data."),
("What is the mean of 2, 4, and 6?", ["4", "3", "6", "12"], 0, "The mean is (2 + 4 + 6) / 3 = 4.")
],
"Math": [
("What is 15% of 200?", ["30", "15", "20", "45"], 0, "15% of 200 is 0.15 × 200 = 30."),
("Solve: x + 7 = 12.", ["x = 5", "x = 19", "x = 7", "x = 4"], 0, "Subtract 7 from both sides: x = 12 − 7 = 5."),
("What is the probability of heads on a fair coin toss?", ["1/2", "1/3", "1", "0"], 0, "There are two equally likely outcomes and one is heads, so the probability is 1/2.")
],
"Art": [
("Which art element describes how light or dark an area appears?", ["Value", "Tempo", "Rhythm", "Pitch"], 0, "Value describes the lightness or darkness of a visual element."),
("In visual design, what does contrast help create?", ["A noticeable difference between elements", "A list of historical dates", "A musical beat", "A mathematical equation"], 0, "Contrast uses differences, such as light versus dark, to help elements stand apart."),
("What does composition mean in art?", ["The arrangement of elements in an artwork", "The speed of a song", "The age of an object", "The number of spreadsheet rows"], 0, "Composition is the way visual parts are organized in an artwork.")
],
"History": [
("What is a primary source in history?", ["Evidence created during or close to the period studied", "A modern summary only", "A fictional story with no evidence", "A random list of dates"], 0, "Letters, diaries, photographs, objects, and records from the period can be primary sources."),
("Why compare different historical sources?", ["To check evidence and identify different perspectives", "To make every source identical", "To avoid evidence", "To replace facts with guesses"], 0, "Sources may contain different viewpoints, gaps, or biases."),
("Which question helps evaluate a historical source?", ["Who created it, and for what purpose?", "Is it the longest document?", "Does it use bright colours?", "Can it be turned into a song?"], 0, "Creator and purpose help explain context and possible bias.")
],
"Music": [
("What is tempo in music?", ["The speed of the music", "The surface of a painting", "The age of a composer", "The loudness of one note only"], 0, "Tempo describes how fast or slow music is performed."),
("What is melody?", ["A sequence of notes heard as a musical line", "The speed of a computer", "The arrangement of a painting", "A historical document"], 0, "Melody is a sequence of pitches often recognized as the tune."),
("What does timbre help a listener distinguish?", ["The sound quality of different instruments or voices", "The pages in a book", "The brightness of a painting", "A maths result"], 0, "Timbre is the tone quality that can make instruments sound different.")
]
}

TOPICS = ["AI", "Data Science", "Math", "Art", "History", "Music"]
available_topics = sorted(df["topic"].dropna().unique().tolist())
default_topic = "Data Science" if "Data Science" in available_topics else available_topics[0]

page = st.sidebar.radio("Navigate", ["Home & Recommendations", "Video Lessons", "Text Lessons", "Take a Quiz"])

if page == "Home & Recommendations":
    st.header("Find a course for you")
    c1, c2, c3 = st.columns(3)
    with c1:
        topic = st.selectbox("Topic", available_topics, index=available_topics.index(default_topic))
    with c2:
        difficulty = st.selectbox("Difficulty", ["Any"] + sorted(df["difficulty"].unique().tolist()))
    with c3:
        content_type = st.selectbox("Content type", ["Any"] + sorted(df["content_type"].unique().tolist()))
    count = st.slider("Number of recommendations", 1, 10, 5)
    if st.button("Recommend Courses", type="primary"):
        filtered = df[df["topic"] == topic].copy()
        if difficulty != "Any":
            filtered = filtered[filtered["difficulty"] == difficulty]
        if content_type != "Any":
            filtered = filtered[filtered["content_type"] == content_type]
        if filtered.empty:
            st.warning("No courses match these preferences. Try Any for difficulty or content type.")
        else:
            features_cols = ["topic", "difficulty", "content_type"]
            if "learning_style" in df.columns:
                features_cols.append("learning_style")
            features = df[features_cols].fillna("").astype(str).agg(" ".join, axis=1)
            matrix = TfidfVectorizer().fit_transform(features)
            similarity = cosine_similarity(matrix)
            ref_index = filtered.index[0]
            ranking = sorted(range(len(df)), key=lambda i: similarity[ref_index][i], reverse=True)
            rows, seen = [], set()
            for i in ranking:
                row = df.iloc[i]
                if row["topic"] != topic or row["course_id"] in seen:
                    continue
                if difficulty != "Any" and row["difficulty"] != difficulty:
                    continue
                if content_type != "Any" and row["content_type"] != content_type:
                    continue
                seen.add(row["course_id"])
                rows.append({"Course ID": row["course_id"], "Course Title": row["title"],
                             "Topic": row["topic"], "Difficulty": row["difficulty"],
                             "Content Type": row["content_type"],
                             "Similarity Score": round(float(similarity[ref_index][i]), 3)})
                if len(rows) >= count:
                    break
            results = pd.DataFrame(rows)
            st.subheader("Recommended Courses")
            st.success(f"Found {len(results)} matching course(s).")
            for _, course in results.iterrows():
                with st.container(border=True):
                    st.subheader("📘 " + str(course["Course Title"]))
                    a, b, c = st.columns(3)
                    a.write(f"**Topic:** {course['Topic']}")
                    b.write(f"**Difficulty:** {course['Difficulty']}")
                    c.write(f"**Format:** {course['Content Type']}")
                    st.caption(f"Course ID: {course['Course ID']} | Similarity: {course['Similarity Score']}")
                    st.write("Use the left menu to open a video lesson, read notes, or take a quiz for this topic.")
            st.download_button("Download Recommendations (CSV)", results.to_csv(index=False).encode("utf-8"),
                               "recommended_courses.csv", "text/csv")
    st.info("The CSV contains course metadata only. Titles are generated labels; lessons and quizzes are sample content in this app.")

elif page == "Video Lessons":
    st.header("▶️ Video Lessons")
    st.write("Choose a topic, then open YouTube search results. You can also paste a specific YouTube video URL to play it here.")
    video_topic = st.selectbox("Topic for videos", TOPICS, index=TOPICS.index(default_topic) if default_topic in TOPICS else 1)
    searches = {
        "AI": "artificial intelligence basics for beginners educational",
        "Data Science": "data science for beginners educational",
        "Math": "mathematics basics lessons educational",
        "Art": "elements of art and design basics educational",
        "History": "world history introduction educational",
        "Music": "music theory basics rhythm melody tempo educational"
    }
    st.subheader(LESSONS[video_topic][0])
    st.link_button("Search YouTube for this topic", "https://www.youtube.com/results?search_query=" + quote_plus(searches[video_topic]))
    url = st.text_input("Optional: paste a YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
    if url.strip():
        if "youtube.com" in url or "youtu.be" in url:
            st.video(url)
        else:
            st.warning("Please enter a YouTube URL.")

elif page == "Text Lessons":
    st.header("📖 Text Lessons")
    lesson_topic = st.selectbox("Choose a topic", TOPICS, index=TOPICS.index(default_topic) if default_topic in TOPICS else 1)
    st.subheader(LESSONS[lesson_topic][0])
    st.markdown(LESSONS[lesson_topic][1])
    st.caption("These are introductory sample notes. You can expand them with your own course materials.")

elif page == "Take a Quiz":
    st.header("📝 Interactive Quizzes")
    quiz_topic = st.selectbox("Choose a quiz topic", TOPICS, index=TOPICS.index(default_topic) if default_topic in TOPICS else 1)
    questions = QUIZZES[quiz_topic]
    st.write(f"This quiz has {len(questions)} questions.")
    with st.form("quiz_" + quiz_topic):
        answers = []
        for i, (question, options, correct_index, explanation) in enumerate(questions):
            st.markdown(f"**Question {i + 1}. {question}**")
            answer = st.radio("Choose one:", options, index=None, key=f"{quiz_topic}_{i}", label_visibility="collapsed")
            answers.append(answer)
        submitted = st.form_submit_button("Submit Quiz", type="primary")
    if submitted:
        score = sum(1 for i, q in enumerate(questions) if answers[i] == q[1][q[2]])
        st.subheader(f"Your score: {score} / {len(questions)}")
        st.progress(score / len(questions))
        if score == len(questions):
            st.success("Excellent! You answered every question correctly.")
        elif score >= 2:
            st.success("Good work! Read the explanations to improve further.")
        else:
            st.info("Keep practising. Review the explanations and try again.")
        for i, (question, options, correct_index, explanation) in enumerate(questions):
            with st.container(border=True):
                st.markdown(f"**Question {i + 1}: {question}**")
                if answers[i] is None:
                    st.warning("You did not answer this question.")
                elif answers[i] == options[correct_index]:
                    st.success("Your answer is correct.")
                else:
                    st.error("Your answer is incorrect.")
                st.write("**Correct answer:** " + options[correct_index])
                st.write("**Explanation:** " + explanation)
    st.caption("Scores are displayed in the current session and are not saved to a database.")

st.divider()
st.caption("Academic project demo | Recommendations use your CSV; lessons and quizzes are sample content.")
