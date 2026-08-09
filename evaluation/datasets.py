from langsmith import Client

DATASET_NAME = "cgu Q&A"

EXAMPLES =[
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
        "outputs": {"answer": "The tuition fee per semester for BTech Computer Science Engineering (Data Science) is ₹1,45,000 for a duration of four years."},
    },
    {
        "inputs": {"question": "What is the tuition fee for BTech Mechanical Engineering?"},
        "outputs": {"answer": "The tuition fee per semester for BTech Mechanical Engineering is ₹1,12,500 and the program duration is four years."},
    },
    {
        "inputs": {"question": "What is the fee structure for lateral entry to BTech Computer Science Engineering?"},
        "outputs": {"answer": "For lateral entry to BTech Computer Science Engineering, the fee is ₹1,12,500 per semester and the duration of the program is three years."},
    },
    # {
    #     "inputs": {"question": "What is the tuition fee per semester for M.Sc programs at CGU?"},
    #     "outputs": {"answer": "The tuition fee for all M.Sc programs at C. V. Raman Global University is ₹40,000 per semester with a duration of two years."},
    # },
    # {
    #     "inputs": {"question": "What is the fee per semester for the MBA program?"},
    #     "outputs": {"answer": "The MBA program has a tuition fee of ₹97,500 per semester and the duration of the program is two years."},
    # },
    # {
    #     "inputs": {"question": "What is the tuition fee for the MA English program?"},
    #     "outputs": {"answer": "The tuition fee for the MA English program is ₹28,000 per semester and the duration of the program is two years."},
    # },
    # {
    #     "inputs": {"question": "What is the fee structure for B.Sc Agriculture?"},
    #     "outputs": {"answer": "The B.Sc Agriculture program has a tuition fee of ₹80,000 per semester and the duration of the program is four years."},
    # },
    # {
    #     "inputs": {"question": "What is the tuition fee for the BBA program?"},
    #     "outputs": {"answer": "The Bachelor of Business Administration (BBA) program has a tuition fee of ₹60,000 per semester with a duration of three years."},
    # },
    # {
    #     "inputs": {"question": "What is the fee per semester for the BCA program?"},
    #     "outputs": {"answer": "The Bachelor of Computer Application (BCA) program has a tuition fee of ₹60,000 per semester and the duration is three years."},
    # },
    # {
    #     "inputs": {"question": "What is the tuition fee for the MCA program?"},
    #     "outputs": {"answer": "The Master of Computer Application (MCA) program has a tuition fee of ₹80,000 per semester and the duration is two years."},
    # },
    # {
    #     "inputs": {"question": "What is the fee per semester for M.Tech programs?"},
    #     "outputs": {"answer": "All M.Tech programs have a tuition fee of ₹63,500 per semester and a duration of two years."},
    # },
    # {
    #     "inputs": {"question": "Is there any stipend available for M.Tech students?"},
    #     "outputs": {"answer": "Yes, M.Tech students can receive a stipend of ₹7,500 per month after each semester if they maintain a minimum CGPA of 7.5 and at least 75% attendance, for a maximum of two years."},
    # },
    # {
    #     "inputs": {"question": "What is the admission fee and tuition fee for Ph.D programs?"},
    #     "outputs": {"answer": "Ph.D programs require an admission fee of ₹20,000 in the first year, a tuition fee of ₹60,000 per year, and a thesis submission fee of ₹20,000 in the final year, with a duration of three years."},
    # },
    # {
    #     "inputs": {"question": "What is the hostel fee for girls staying in non-AC six-sharing rooms?"},
    #     "outputs": {"answer": "The hostel fee for girls staying in non-AC six-sharing rooms is ₹45,000 per academic year."},
    # },
    # {
    #     "inputs": {"question": "What is the hostel fee for boys staying in AC rooms with double occupancy?"},
    #     "outputs": {"answer": "The hostel fee for boys staying in AC rooms with two or one occupancy is ₹1,25,000 per academic year."},
    # },
    # {
    #     "inputs": {"question": "What are the mess charges for hostel students per academic year?"},
    #     "outputs": {"answer": "The mess charges for hostel students are ₹45,000 per academic year."},
    # },
    # {
    #     "inputs": {"question": "Is hostel room rent refundable after classes start?"},
    #     "outputs": {"answer": "No, once enrollment is completed and classes have commenced, the hostel room rent for the academic year is non-refundable."},
    # },
    # {
    #     "inputs": {"question": "What is the refund policy for hostel mess fees?"},
    #     "outputs": {"answer": "Hostel mess fees are refundable after deducting the proportionate amount for the period used and a maintenance charge of ₹5,000, subject to clearance of any outstanding dues."},
    # },
    # {
    #     "inputs": {"question": "What is the transport fee for students commuting from Bhubaneswar?"},
    #     "outputs": {"answer": "The transport fee for students commuting from Bhubaneswar is ₹18,000 per year."},
    # },
    # {
    #     "inputs": {"question": "What is the transport fee for students commuting from Cuttack?"},
    #     "outputs": {"answer": "The transport fee for students commuting from Cuttack is ₹30,000 per year."},
    # },
    # {
    #     "inputs": {"question": "What is the tuition fee for international undergraduate students?"},
    #     "outputs": {"answer": "International undergraduate students are required to pay a tuition fee of 8,000 USD for programs such as BTech, B.Sc (Hons.) Agriculture, and BBA."},
    # },
    # {
    #     "inputs": {"question": "What is the tuition fee for international postgraduate and PhD students?"},
    #     "outputs": {"answer": "International postgraduate, PhD, and exchange program students are required to pay a tuition fee of 4,000 USD."},
    # },
    # {
    #     "inputs": {"question": "What is the hostel and mess fee range for international students?"},
    #     "outputs": {"answer": "For international students, hostel and mess fees range between 1,200 USD and 2,000 USD per annum depending on room type, facilities, and availability."},
    # }
]
def create_dataset():
    client = Client()
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=EXAMPLES
    )
    return DATASET_NAME



