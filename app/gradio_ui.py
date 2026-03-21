"""Gradio interface for credit scoring predictions."""

import gradio as gr

from app.services.prediction_service import predict_credit_default


def make_prediction(
    name_contract_type,
    code_gender,
    flag_own_car,
    name_family_status,
    name_education_type,
    amt_income_total,
    amt_credit,
    amt_annuity,
    amt_goods_price,
    cnt_fam_members,
    days_birth,
    days_employed,
    ext_source_2,
    ext_source_3,
):
    """Run prediction from Gradio inputs."""
    features = {
        "name_contract_type": name_contract_type,
        "code_gender": code_gender,
        "flag_own_car": flag_own_car,
        "name_family_status": name_family_status,
        "name_education_type": name_education_type,
        "amt_income_total": float(amt_income_total),
        "amt_credit": float(amt_credit),
        "amt_annuity": float(amt_annuity),
        "amt_goods_price": float(amt_goods_price) if amt_goods_price else None,
        "cnt_fam_members": float(cnt_fam_members),
        "days_birth": int(days_birth),
        "days_employed": int(days_employed),
        "ext_source_2": float(ext_source_2) if ext_source_2 else None,
        "ext_source_3": float(ext_source_3) if ext_source_3 else None,
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


with gr.Blocks(title="Credit Scoring") as demo:
    gr.Markdown("# Credit Scoring - Loan Default Prediction")
    gr.Markdown("Enter the applicant's information to predict loan repayment.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Loan Information")
            name_contract_type = gr.Dropdown(
                choices=["Cash loans", "Revolving loans"],
                label="Contract Type",
                value="Cash loans",
            )
            amt_income_total = gr.Number(label="Annual Income", value=202500.0)
            amt_credit = gr.Number(label="Credit Amount", value=406597.5)
            amt_annuity = gr.Number(label="Annuity Amount", value=24700.5)
            amt_goods_price = gr.Number(label="Goods Price", value=351000.0)

        with gr.Column():
            gr.Markdown("### Personal Information")
            code_gender = gr.Dropdown(choices=["M", "F"], label="Gender", value="F")
            flag_own_car = gr.Dropdown(choices=["Y", "N"], label="Owns Car?", value="N")
            name_family_status = gr.Dropdown(
                choices=["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
                label="Family Status",
                value="Married",
            )
            name_education_type = gr.Dropdown(
                choices=["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
                label="Education Level",
                value="Secondary / secondary special",
            )
            cnt_fam_members = gr.Number(label="Family Members", value=2.0, precision=0)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Time-based Features")
            days_birth = gr.Number(label="Days Birth (negative, e.g. -9461 = ~26 years)", value=-9461)
            days_employed = gr.Number(label="Days Employed (negative, e.g. -637 = ~1.7 years)", value=-637)

        with gr.Column():
            gr.Markdown("### External Scores")
            ext_source_2 = gr.Number(label="External Source 2 (0-1)", value=0.263)
            ext_source_3 = gr.Number(label="External Source 3 (0-1)", value=0.139)

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
            amt_income_total,
            amt_credit,
            amt_annuity,
            amt_goods_price,
            cnt_fam_members,
            days_birth,
            days_employed,
            ext_source_2,
            ext_source_3,
        ],
        outputs=output,
    )
