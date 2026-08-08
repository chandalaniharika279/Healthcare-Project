import streamlit as st
import re
from symptom_knowledge import infer_from_dataset
from adaptive_questioning import next_question
from severity_utils import severity_to_band

# =====================================================
# UI CONFIG
# =====================================================
st.set_page_config(
    page_title="Healthcare Triage",
    layout="centered"
)

st.title("🩺 Intelligent Healthcare Triage System")

# =====================================================
# AUTOMATIC UI NORMALIZATION (NO MANUAL LISTING)
# =====================================================
MEDICAL_SUFFIXES = [
    "pain", "infection", "swelling", "injury",
    "breath", "bleeding", "fracture", "fever",
    "vomiting", "nausea", "headache", "trauma"
]

MANUAL_MAP = {
    "shortnessofbreath": "shortness of breath"
}

def pretty(symptom: str) -> str:
    if symptom in MANUAL_MAP:
        return MANUAL_MAP[symptom]

    symptom = symptom.replace("-", " ").replace("_", " ")
    symptom = re.sub(r"([a-z])([A-Z])", r"\1 \2", symptom)
    symptom = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", symptom)

    for suf in MEDICAL_SUFFIXES:
        symptom = re.sub(f"(\\w+)({suf})$", r"\1 \2", symptom)

    return symptom.lower().strip()

# =====================================================
# CONSTANTS
# =====================================================
EMERGENCY_THRESHOLD = 0.7

# =====================================================
# SESSION STATE
# =====================================================
for k, v in {
    "conversation": "",
    "asked": set(),
    "matched": None,
    "finished": False,
    "other_checked": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================
# INPUT
# =====================================================
text = st.text_input("Enter symptoms")

if st.button("Analyze"):
    st.session_state.conversation = text.lower().strip()
    st.session_state.asked.clear()
    st.session_state.finished = False
    st.session_state.other_checked = False
    st.rerun()

# =====================================================
# MAIN ASSESSMENT
# =====================================================
if st.session_state.conversation and not st.session_state.finished:

    severity, emergency, matched = infer_from_dataset(
        st.session_state.conversation
    )
    st.session_state.matched = matched

    st.subheader("🔍 Current Assessment")
    st.write(
        f"Severity Score: {severity} "
        f"({severity_to_band(severity)})"
    )

    # 🚨 EMERGENCY POP-UP
    if emergency >= EMERGENCY_THRESHOLD:
        st.error(
            "🚨 EMERGENCY DETECTED\n\n"
            "Your symptoms indicate a high emergency risk.\n"
            "Please seek immediate medical attention or "
            "contact emergency services."
        )
        st.stop()
    else:
        st.success("🟢 Emergency Risk: LOW")

    # =================================================
    # ADAPTIVE QUESTIONING
    # =================================================
    if not matched.empty:
        question = next_question(
            matched, st.session_state.asked
        )
    else:
        question = None

    if question:
        ans = st.radio(question, ["yes", "no"])
        if st.button("Confirm"):
            symptom = (
                question.replace("Do you have ", "")
                .replace("?", "")
                .replace(" ", "")
            )
            st.session_state.asked.add(symptom)
            if ans == "yes":
                st.session_state.conversation += " " + symptom
            st.rerun()
    else:
        st.session_state.finished = True
        st.rerun()

# =====================================================
# OTHER SYMPTOMS
# =====================================================
if st.session_state.finished and not st.session_state.other_checked:
    other = st.radio("Any other symptoms?", ["no", "yes"])
    if other == "yes":
        extra = st.text_input("Describe other symptoms")
        if st.button("Submit"):
            st.session_state.conversation += (
                " " + extra.lower()
            )
            st.session_state.other_checked = True
            st.rerun()
    else:
        st.session_state.other_checked = True
        st.rerun()

# =====================================================
# FINAL DECISION
# =====================================================
if st.session_state.other_checked:
    sev, emer, _ = infer_from_dataset(
        st.session_state.conversation
    )

    st.subheader("✅ Final Decision")

    if emer >= EMERGENCY_THRESHOLD:
        st.error("🚨 Seek emergency care immediately.")
    elif sev >= 0.6:
        st.warning("⚠️ Consult a doctor soon.")
    else:
        st.success("🟢 Home care advised.")

    # =================================================
    # EXPLANATION
    # =================================================
    st.subheader("🧠 Explanation")
    st.write("Symptoms considered:")

    for s in st.session_state.asked:
        st.write(f"- {pretty(s)}")

    st.caption(
        f"Final Severity: {sev} "
        f"({severity_to_band(sev)}) | "
        f"Emergency: {emer}"
    )
