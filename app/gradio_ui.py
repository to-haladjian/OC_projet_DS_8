"""Gradio interface for credit scoring predictions."""

from datetime import date, timedelta

import gradio as gr

from app.services.prediction_service import predict_credit_default


def make_prediction(
    name_contract_type,
    code_gender,
    flag_own_car,
    name_family_status,
    name_education_type,
    organization_type,
    amt_income_total,
    amt_credit,
    amt_annuity,
    amt_goods_price,
    cnt_fam_members,
    birth_date,
    employment_start_date,
    registration_date,
    id_publish_date,
    last_phone_change_date,
    ext_source_2,
    ext_source_3,
    region_population_relative,
    region_rating_client_w_city,
    reg_city_not_live_city,
    floorsmax_avg,
    totalarea_mode,
    years_beginexpluatation_medi,
    obs_30_cnt_social_circle,
    def_30_cnt_social_circle,
    amt_req_credit_bureau_qrt,
    flag_document_3,
):
    """Run prediction from Gradio inputs."""
    features = {
        "name_contract_type": name_contract_type,
        "code_gender": code_gender,
        "flag_own_car": flag_own_car,
        "name_family_status": name_family_status,
        "name_education_type": name_education_type,
        "organization_type": organization_type or None,
        "amt_income_total": float(amt_income_total),
        "amt_credit": float(amt_credit),
        "amt_annuity": float(amt_annuity),
        "amt_goods_price": float(amt_goods_price) if amt_goods_price is not None else None,
        "cnt_fam_members": float(cnt_fam_members),
        # Real-world dates; preprocessing converts them to the model's
        # "days relative to today" convention. None = unset (unemployed for jobs).
        "birth_date": birth_date,
        "employment_start_date": employment_start_date,
        "registration_date": registration_date,
        "id_publish_date": id_publish_date,
        "last_phone_change_date": last_phone_change_date,
        "ext_source_2": float(ext_source_2) if ext_source_2 is not None else None,
        "ext_source_3": float(ext_source_3) if ext_source_3 is not None else None,
        "region_population_relative": float(region_population_relative) if region_population_relative is not None else None,
        "region_rating_client_w_city": int(region_rating_client_w_city) if region_rating_client_w_city is not None else None,
        "reg_city_not_live_city": int(reg_city_not_live_city) if reg_city_not_live_city is not None else None,
        "floorsmax_avg": float(floorsmax_avg) if floorsmax_avg is not None else None,
        "totalarea_mode": float(totalarea_mode) if totalarea_mode is not None else None,
        "years_beginexpluatation_medi": float(years_beginexpluatation_medi) if years_beginexpluatation_medi is not None else None,
        "obs_30_cnt_social_circle": float(obs_30_cnt_social_circle) if obs_30_cnt_social_circle is not None else None,
        "def_30_cnt_social_circle": float(def_30_cnt_social_circle) if def_30_cnt_social_circle is not None else None,
        "amt_req_credit_bureau_qrt": float(amt_req_credit_bureau_qrt) if amt_req_credit_bureau_qrt is not None else None,
        "flag_document_3": int(flag_document_3) if flag_document_3 is not None else None,
    }

    _, default_probability, credit_approved = predict_credit_default(features)

    decision = "APPROVED" if credit_approved else "DENIED"
    color = "green" if credit_approved else "red"

    result = (
        f"## Decision: <span style='color:{color}'>{decision}</span>\n\n"
        f"**Default Probability:** {default_probability:.2%}\n\n"
        f"**Threshold:** 8.74%\n\n"
        f"**Credit Approved:** {'Yes' if credit_approved else 'No'}"
    )
    return result


# All organization types from training data, ordered by frequency
_ORGANIZATION_TYPES = [
    None,
    "Business Entity Type 3",
    "XNA",
    "Self-employed",
    "Other",
    "Medicine",
    "Business Entity Type 2",
    "Government",
    "School",
    "Trade: type 7",
    "Kindergarten",
    "Construction",
    "Business Entity Type 1",
    "Transport: type 4",
    "Trade: type 3",
    "Industry: type 9",
    "Industry: type 3",
    "Security",
    "Housing",
    "Industry: type 11",
    "Military",
    "Bank",
    "Agriculture",
    "Police",
    "Transport: type 2",
    "Postal",
    "Security Ministries",
    "Trade: type 2",
    "Restaurant",
    "Services",
    "University",
    "Industry: type 7",
    "Transport: type 3",
    "Industry: type 1",
    "Hotel",
    "Electricity",
    "Industry: type 4",
    "Trade: type 6",
    "Industry: type 5",
    "Insurance",
    "Telecom",
    "Emergency",
    "Industry: type 2",
    "Advertising",
    "Realtor",
    "Culture",
    "Industry: type 12",
    "Trade: type 1",
    "Mobile",
    "Legal Services",
    "Cleaning",
    "Transport: type 1",
    "Industry: type 6",
    "Industry: type 10",
    "Religion",
    "Industry: type 13",
    "Trade: type 4",
    "Trade: type 5",
    "Industry: type 8",
]

with gr.Blocks(title="Credit Scoring") as demo:
    gr.Markdown("# Credit Scoring - Loan Default Prediction")
    gr.Markdown(
        "Enter the applicant's information to predict loan repayment risk."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Loan Information")
            name_contract_type = gr.Dropdown(
                choices=["Cash loans", "Revolving loans"],
                label="Contract Type",
                value="Cash loans",  # 90.5% of training data
            )
            amt_income_total = gr.Number(
                label="Annual Income",
                value=147150.0,
            )
            amt_credit = gr.Number(
                label="Credit Amount",
                value=513531.0,
            )
            amt_annuity = gr.Number(
                label="Annuity Amount",
                value=24903.0,
            )
            amt_goods_price = gr.Number(
                label="Goods Price — optional",
                value=450000.0,
            )

        with gr.Column():
            gr.Markdown("### Personal Information")
            code_gender = gr.Dropdown(
                choices=["F", "M"],
                label="Gender",
                value="F",
            )
            flag_own_car = gr.Dropdown(
                choices=["N", "Y"],
                label="Owns Car?",
                value="N",
            )
            name_family_status = gr.Dropdown(
                choices=[
                    "Married",               # 64%
                    "Single / not married",  # 15%
                    "Civil marriage",        # 10%
                    "Separated",             # 6%
                    "Widow",                 # 5%
                ],
                label="Family Status",
                value="Married",
            )
            name_education_type = gr.Dropdown(
                choices=[
                    "Secondary / secondary special",  # 71%
                    "Higher education",               # 24%
                    "Incomplete higher",              # 3%
                    "Lower secondary",                # 1%
                    "Academic degree",                # <1%
                ],
                label="Education Level",
                value="Secondary / secondary special",
            )
            cnt_fam_members = gr.Number(
                label="Family Members",
                value=2.0,
                precision=0,
            )
            organization_type = gr.Dropdown(
                choices=_ORGANIZATION_TYPES,
                label="Organization Type — optional (most common: Business Entity Type 3)",
                value="Business Entity Type 3",
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Key Dates")
            _today = date.today()
            birth_date = gr.DateTime(
                label="Date of Birth",
                include_time=False,
                type="datetime",
                value=(_today - timedelta(days=15750)).isoformat(),
            )
            employment_start_date = gr.DateTime(
                label="Employment Start Date — leave empty if unemployed/retired",
                include_time=False,
                type="datetime",
                value=(_today - timedelta(days=1213)).isoformat(),
            )
            registration_date = gr.DateTime(
                label="Registration Date — optional",
                include_time=False,
                type="datetime",
                value=(_today - timedelta(days=4504)).isoformat(),
            )
            id_publish_date = gr.DateTime(
                label="ID Issue Date — optional",
                include_time=False,
                type="datetime",
                value=(_today - timedelta(days=3254)).isoformat(),
            )
            last_phone_change_date = gr.DateTime(
                label="Last Phone Change Date — optional",
                include_time=False,
                type="datetime",
                value=(_today - timedelta(days=757)).isoformat(),
            )

        with gr.Column():
            gr.Markdown("### External Scores")
            ext_source_2 = gr.Number(
                label="External Source 2 — 0 to 1, optional",
                value=0.566,
            )
            ext_source_3 = gr.Number(
                label="External Source 3 — 0 to 1, optional",
                value=0.535,
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Region & Housing — optional")
            region_population_relative = gr.Number(
                label="Region Population Relative",
                value=0.0188,
            )
            region_rating_client_w_city = gr.Dropdown(
                choices=[None, 1, 2, 3],
                label="Region Rating w/ City",
                value=2,
            )
            reg_city_not_live_city = gr.Dropdown(
                choices=[None, 0, 1],
                label="Registration City ≠ Living City",
                value=0,
            )
            floorsmax_avg = gr.Number(
                label="Max Floors Avg — 0 to 1",
                value=0.1667,
            )
            totalarea_mode = gr.Number(
                label="Total Area Mode — 0 to 1",
                value=0.0688,
            )
            years_beginexpluatation_medi = gr.Number(
                label="Years Building Exploitation Median — 0 to 1",
                value=0.9816,
            )

        with gr.Column():
            gr.Markdown("### Social Circle & Documents — optional")
            obs_30_cnt_social_circle = gr.Number(
                label="Observable 30-DPD in Social Circle",
                value=0.0,
            )
            def_30_cnt_social_circle = gr.Number(
                label="Defaults 30-DPD in Social Circle",
                value=0.0,
            )
            amt_req_credit_bureau_qrt = gr.Number(
                label="Credit Bureau Enquiries — last quarter",
                value=0.0,
            )
            flag_document_3 = gr.Dropdown(
                choices=[None, 0, 1],
                label="Document 3 Provided (1=yes)",
                value=1,
            )

    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.Markdown(label="Prediction Result")

    predict_btn.click(
        fn=make_prediction,
        inputs=[
            name_contract_type,
            code_gender,
            flag_own_car,
            name_family_status,
            name_education_type,
            organization_type,
            amt_income_total,
            amt_credit,
            amt_annuity,
            amt_goods_price,
            cnt_fam_members,
            birth_date,
            employment_start_date,
            registration_date,
            id_publish_date,
            last_phone_change_date,
            ext_source_2,
            ext_source_3,
            region_population_relative,
            region_rating_client_w_city,
            reg_city_not_live_city,
            floorsmax_avg,
            totalarea_mode,
            years_beginexpluatation_medi,
            obs_30_cnt_social_circle,
            def_30_cnt_social_circle,
            amt_req_credit_bureau_qrt,
            flag_document_3,
        ],
        outputs=output,
    )
