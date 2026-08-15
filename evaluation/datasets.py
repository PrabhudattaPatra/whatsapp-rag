from langsmith import Client

# v2: the original "cgu Q&A" dataset's ground-truth answers were stale --
# data/fee_structure.md has been revised (branch fees, hostel fees,
# transport fees, M.Tech stipend removed, international-student section
# removed entirely) since those examples were written, which is exactly
# why the correctness evaluator scored ~0.32 despite groundedness/
# relevance/retrieval_relevance all scoring 0.9+. Every answer below was
# re-derived directly from the current data/fee_structure.md (read in
# full, not guessed) so ground truth matches what's actually ingested.
# Using a new dataset name rather than overwriting "cgu Q&A" in place --
# LangSmith example sync here is add-missing-by-question-text, so
# reusing the old name wouldn't update already-stale remote examples,
# and a fresh name keeps historical experiments against the old (stale)
# dataset interpretable on their own terms rather than silently rewritten.
DATASET_NAME = "cgu Q&A v2"

EXAMPLES = [
    {
        "inputs": {"question": "How can I pay my fees online at C. V. Raman Global University?"},
        "outputs": {"answer": "Fees can be paid online by visiting the Eduqfix payment portal, selecting C. V. Raman Global University as the branch, entering the university roll number or registration number, choosing the fee type, entering the amount, and clicking Continue to complete the payment."},
    },
    {
        "inputs": {"question": "What is the tuition fee per semester for BTech Computer Science Engineering (AI & Machine Learning)?"},
        "outputs": {"answer": "The tuition fee per semester for BTech Computer Science Engineering (AI & Machine Learning) is ₹1,50,000 and the duration of the program is four years."},
    },
    {
        "inputs": {"question": "What is the fee per semester for BTech Computer Science Engineering (Data Science)?"},
        "outputs": {"answer": "The tuition fee per semester for BTech Computer Science Engineering (Data Science) is ₹1,50,000 for a duration of four years."},
    },
    {
        "inputs": {"question": "What is the tuition fee for BTech Mechanical Engineering?"},
        "outputs": {"answer": "The tuition fee per semester for BTech Mechanical Engineering is ₹1,12,500 and the program duration is four years."},
    },
    {
        "inputs": {"question": "What is the tuition fee per semester for BTech Electronics & Communication Engineering?"},
        "outputs": {"answer": "The tuition fee per semester for BTech Electronics & Communication Engineering is ₹1,27,500 and the program duration is four years."},
    },
    {
        "inputs": {"question": "What is the fee structure for lateral entry to BTech Computer Science Engineering (AI & Machine Learning)?"},
        "outputs": {"answer": "For lateral entry to BTech Computer Science Engineering (AI & Machine Learning), the fee is ₹1,20,000 per semester and the duration of the program is three years."},
    },
    {
        "inputs": {"question": "What is the lateral entry fee for BTech Mechanical Engineering?"},
        "outputs": {"answer": "For lateral entry to BTech Mechanical Engineering, the fee is ₹97,500 per semester and the duration of the program is three years."},
    },
    {
        "inputs": {"question": "What is the fee per semester for M.Tech programs?"},
        "outputs": {"answer": "All M.Tech programs have a tuition fee of ₹63,500 per semester and a duration of two years, with no stipend."},
    },
    {
        "inputs": {"question": "Is there any stipend available for M.Tech students?"},
        "outputs": {"answer": "No, M.Tech programs at C. V. Raman Global University do not offer a stipend."},
    },
    {
        "inputs": {"question": "What is the fee per semester for the BCA program?"},
        "outputs": {"answer": "The Bachelor in Computer Application (BCA) program has a tuition fee of ₹70,000 per semester and the duration is three years."},
    },
    {
        "inputs": {"question": "What is the tuition fee for the MCA program?"},
        "outputs": {"answer": "The Master in Computer Application (MCA) program has a tuition fee of ₹85,000 per semester and the duration is two years."},
    },
    {
        "inputs": {"question": "What is the tuition fee for the BBA program?"},
        "outputs": {"answer": "The Bachelor in Business Administration (BBA) program has a tuition fee of ₹70,000 per semester with a duration of three years."},
    },
    {
        "inputs": {"question": "What is the fee per semester for the MBA program?"},
        "outputs": {"answer": "The Master in Business Administration (MBA) program has a tuition fee of ₹1,02,500 per semester and the duration of the program is two years."},
    },
    {
        "inputs": {"question": "What is the fee structure for B.Sc Agriculture?"},
        "outputs": {"answer": "The B.Sc Agriculture program has a tuition fee of ₹85,000 per semester and the duration of the program is four years."},
    },
    {
        "inputs": {"question": "What is the tuition fee for the B.Pharm program?"},
        "outputs": {"answer": "The B.Pharm program has a tuition fee of ₹80,000 per semester and the duration of the program is four years."},
    },
    {
        "inputs": {"question": "What is the tuition fee for the MA English program?"},
        "outputs": {"answer": "The tuition fee for the MA English program is ₹35,000 per semester and the duration of the program is two years."},
    },
    {
        "inputs": {"question": "What is the tuition fee per semester for M.Sc programs at CGU?"},
        "outputs": {"answer": "The tuition fee for all M.Sc programs at C. V. Raman Global University is ₹40,000 per semester with a duration of two years."},
    },
    {
        "inputs": {"question": "What is the admission fee and tuition fee for Ph.D programs?"},
        "outputs": {"answer": "Ph.D programs require an admission fee of ₹20,000 in the first year, a tuition fee of ₹60,000 per year, and a thesis submission fee of ₹35,000 in the final year."},
    },
    {
        "inputs": {"question": "What is the fee per semester for Diploma programs?"},
        "outputs": {"answer": "All Diploma programs have a tuition fee of ₹30,000 per semester and the duration is three years."},
    },
    {
        "inputs": {"question": "What is the transport fee for students commuting from Bhubaneswar?"},
        "outputs": {"answer": "The transport fee for students commuting from Bhubaneswar is ₹22,000 per year."},
    },
    {
        "inputs": {"question": "What is the transport fee for students commuting from Cuttack?"},
        "outputs": {"answer": "The transport fee for students commuting from Cuttack is ₹32,000 per year."},
    },
    {
        "inputs": {"question": "What is the hostel fee for boys staying in AC rooms with single or double occupancy?"},
        "outputs": {"answer": "The hostel fee for boys in AC rooms with 01 or 02 bedded occupancy is ₹1,25,000 per academic year."},
    },
    {
        "inputs": {"question": "What is the hostel fee for girls staying in non-AC six-sharing rooms?"},
        "outputs": {"answer": "The hostel fee for girls staying in non-AC six-bedded rooms is ₹45,000 per academic year."},
    },
    {
        "inputs": {"question": "What are the mess charges for hostel students per academic year?"},
        "outputs": {"answer": "Mess/fooding charges are ₹48,000 for 10 months, or ₹57,600 for 12 months for students staying during vacation."},
    },
    {
        "inputs": {"question": "Is hostel room rent refundable after classes start?"},
        "outputs": {"answer": "No, once enrollment is completed and classes have commenced, the hostel room rent for the academic year is non-refundable, even if the hostel premises are not used."},
    },
]


def create_dataset():
    client = Client()
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=EXAMPLES
    )
    return DATASET_NAME
