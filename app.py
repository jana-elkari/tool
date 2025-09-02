from flask import Flask, render_template, request, session, redirect, url_for
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import secrets
import os

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# === Files ===
CSV_FILE = 'users.csv'
CAUSAL_EFFECTS_CSV = 'model_outputs/causal_effects.csv'

# === Load datasets once ===
global_df = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame()
causal_df = pd.read_csv(CAUSAL_EFFECTS_CSV, encoding='utf-8-sig') if os.path.exists(CAUSAL_EFFECTS_CSV) else pd.DataFrame()
print("Loaded users.csv rows:", len(global_df))
print("Loaded causal_effects.csv rows:", len(causal_df))

# === Country → Continent mapping ===
COUNTRY_TO_CONTINENT = {
    "Argentina": "South America",
    "Australia": "Oceania",
    "Austria": "Europe",
    "Canada": "North America",
    "Estonia": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Netherlands": "Europe",
    "Nigeria": "Africa",
    "Norway": "Europe",
    "Pakistan": "Asia",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Spain": "Europe",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "Turkey": "Europe",
    "United Kingdom": "Europe",
    "United States": "North America"
}

# === Collaborative recommender ===
def get_collaborative_recommendation(df, current_user_response):
    user_continent = COUNTRY_TO_CONTINENT.get(current_user_response['country'], None)
    if not user_continent:
        return [f"No continent mapping for '{current_user_response['country']}'."], pd.DataFrame()

    team_size = current_user_response['team_size']
    min_size, max_size = team_size * 0.75, team_size * 1.25

    df = df.copy()
    df['continent'] = df['country'].map(COUNTRY_TO_CONTINENT)

    peers = df[
        (df['continent'] == user_continent) &
        (df['industry'] == current_user_response['industry']) &
        (df['team_size'].astype(float).between(min_size, max_size))
    ]

    if peers.empty:
        return [f"No peer companies in {user_continent}, {current_user_response['industry']}."], pd.DataFrame()

    avg_customer_rel = peers['team_customer_relationship'].astype(float).mean()
    avg_team_comm = peers['team_communication'].astype(float).mean()

    advice = []
    if current_user_response['team_customer_relationship'] <= avg_customer_rel:
        advice.append(f"Your team-customer relationship ({current_user_response['team_customer_relationship']}) "
                      f"is below peers ({avg_customer_rel:.2f}).")
    else:
        advice.append(f"Your team-customer relationship ({current_user_response['team_customer_relationship']}) "
                      f"is above peers ({avg_customer_rel:.2f}).")

    if current_user_response['team_communication'] <= avg_team_comm:
        advice.append(f"Your internal communication ({current_user_response['team_communication']}) "
                      f"is below peers ({avg_team_comm:.2f}).")
    else:
        advice.append(f"Your internal communication ({current_user_response['team_communication']}) "
                      f"is above peers ({avg_team_comm:.2f}).")

    return advice, peers

# === Causal recommender ===
def get_causal_recommendation(user_response, causal_df):
    if causal_df.empty:
        return None, None

    problems = [user_response.get('top1'), user_response.get('top2'),
                user_response.get('top3'), user_response.get('top4'), user_response.get('top5')]
    recs, edges = [], []

    for problem in problems:
        if not problem:
            continue

        matches = causal_df[causal_df['Problem'] == problem]
        if matches.empty:
            recs.append(f"No causal recommendations for '{problem}'.")
            continue

        for _, row in matches.iterrows():
            treatment = row['Treatment']
            effect = row['EffectSize']
            percentage = abs(effect) * 100

            if effect < 0:
                recs.append(f"Improving {treatment} is likely to DECREASE '{problem}' (strength {percentage:.1f}%).")
            else:
                recs.append(f"Improving {treatment} is likely to INCREASE '{problem}' (strength {percentage:.1f}%).")

            edges.append((problem, treatment, effect))

    return recs, edges

def generate_all_plots(filtered_df, user_response):
    plots = {}
    columns_to_plot = [
        'country', 'team_size', 'industry', 'external_partners', 'project_distribution',
        'work', 'role', 'team_customer_relationship', 'team_communication',
        'communication_with_other_teams', 'project_method', 'meeting_frequency',
        'user_feedback_frequency', 'requirements_elicitation', 'elicitation_techniques',
        'assumption', 'doc_willing', 'doc_unwilling_but_produce', 'doc_dont_produce',
        'non_functional_requirements', 'explicit_distinction', 'failed_documentation',
        'requirements_documentation', 'no_documentation_reasons', 'common_problem',
        'top2', 'top3', 'top4', 'top5'
    ]

    for column in columns_to_plot:
        # handle multi-select questions (split on "/")
        if column in ["elicitation_techniques", "explicit_distinction",
                      "requirements_documentation", "no_documentation_reasons",
                      "non_functional_requirements", "requirements_testing_alignment"]:
            all_values = []
            for entry in filtered_df[column].dropna():
                all_values.extend([p.strip() for p in str(entry).split("/") if p.strip()])
            values = pd.Series(all_values).value_counts().reset_index()
            values.columns = [column, "count"]
        else:
            values = filtered_df[column].value_counts().reset_index()
            values.columns = [column, "count"]

        # highlight user answer if available
        user_answer = str(user_response.get(column))
        values["selected"] = values[column].astype(str) == user_answer

        # Plotly bar chart
        fig = px.bar(values, x=column, y="count",
                     title=column.replace("_", " ").title(),
                     color="selected",
                     color_discrete_map={True: "yellow", False: "blue"})

        fig.update_layout(
            showlegend=False,
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=40, b=80)
        )

        plots[column] = fig.to_html(full_html=False, include_plotlyjs=False)

    return plots


# === Routes ===
@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    global global_df

    industries = sorted(global_df['industry'].dropna().unique()) if not global_df.empty else []
    countries = sorted(global_df['country'].dropna().unique()) if not global_df.empty else []

    if request.method == 'POST':
        response = {
            'country': request.form['country'],
            'team_size': int(request.form['team_size']),
            'industry': request.form['industry'],
            'external_partners': request.form['external_partners'],
            'project_distribution': request.form['project_distribution'],
            'work': request.form['work'],
            'role': request.form['role'],
            'team_customer_relationship': int(request.form['team_customer_relationship']),
            'failed_documentation': request.form['failed_documentation'],
            'team_communication': int(request.form['team_communication']),
            'communication_with_other_teams': int(request.form['communication_with_other_teams']),
            'project_method': request.form['project_method'],
            'meeting_frequency': request.form['meeting_frequency'],
            'user_feedback_frequency': request.form['user_feedback_frequency'],
            'assumption': request.form['assumption'],
            'common_problem': request.form['common_problem'],
            'requirements_elicitation': request.form['requirements_elicitation'],
            'doc_willing': request.form['doc_willing'],
            'doc_unwilling_but_produce': request.form['doc_unwilling_but_produce'],
            'doc_dont_produce': request.form['doc_dont_produce'],
            'requirements_testing_alignment': " / ".join(request.form.getlist('requirements_testing_alignment')),
            'elicitation_techniques': " / ".join(request.form.getlist('elicitation_techniques')),
            'explicit_distinction': " / ".join(request.form.getlist('explicit_distinction')),
            'requirements_documentation': " / ".join(request.form.getlist('requirements_documentation')),
            'no_documentation_reasons': " / ".join(request.form.getlist('no_documentation_reasons')),
            'non_functional_requirements': " / ".join(request.form.getlist('non_functional_requirements')),
            'top2': request.form['top2'],
            'top3': request.form['top3'],
            'top4': request.form['top4'],
            'top5': request.form['top5']
        }

        session['user_response'] = response
        global_df = pd.concat([global_df, pd.DataFrame([response])], ignore_index=True)
        global_df.to_csv(CSV_FILE, index=False)

        return redirect(url_for('results'))

    return render_template('survey.html', countries=countries, industries=industries)

@app.route('/results')
def results():
    user_response = session.get('user_response')
    if not user_response:
        return redirect(url_for('survey'))

    # filters
    filtered_df = global_df.copy()
    if (fc := request.args.get('filter_country')): filtered_df = filtered_df[filtered_df['country'] == fc]
    if (fi := request.args.get('filter_industry')): filtered_df = filtered_df[filtered_df['industry'] == fi]
    if (fr := request.args.get('filter_role')): filtered_df = filtered_df[filtered_df['role'] == fr]
    if (fm := request.args.get('filter_method')): filtered_df = filtered_df[filtered_df['project_method'] == fm]
    if (fp := request.args.get('filter_partners')): filtered_df = filtered_df[filtered_df['external_partners'] == fp]
    if (fd := request.args.get('filter_distribution')): filtered_df = filtered_df[filtered_df['project_distribution'] == fd]

    causal_recs, causal_edges = get_causal_recommendation(user_response, causal_df)
    collab_recs, peers_df = get_collaborative_recommendation(filtered_df, user_response)

    recommendation = causal_recs if causal_recs else []
    statistics = collab_recs if collab_recs else []

    # Sankey diagram
    sankey_html = None
    if causal_edges:
        problems = list({p for p, t, e in causal_edges})
        treatments = list({t for p, t, e in causal_edges})
        labels = problems + treatments
        label_to_id = {label: i for i, label in enumerate(labels)}

        sources = [label_to_id[p] for p, t, e in causal_edges]
        targets = [label_to_id[t] for p, t, e in causal_edges]
        values = [abs(e) * 100 for p, t, e in causal_edges]
        colors = ["lightgreen" if e < 0 else "lightcoral" for p, t, e in causal_edges]

        fig = go.Figure()
        fig.add_trace(go.Sankey(
            node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5),
                      label=labels, align="left", color="lightblue"),
            link=dict(source=sources, target=targets, value=values, color=colors)
        ))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=15, color="lightgreen"), name="Decrease problem"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=15, color="lightcoral"), name="Increase problem"))
        fig.update_layout(title_text="Causal Recommender Alluvial Diagram",
                          font=dict(size=12, color="black"),
                          margin=dict(r=200, t=50, b=50, l=50),
                          legend=dict(x=1.05, y=1, bgcolor="rgba(255,255,255,0.7)",
                                      bordercolor="black", borderwidth=1))
        sankey_html = fig.to_html(full_html=False)

    # Generate Plotly plots
    plots = generate_all_plots(filtered_df, user_response)

    return render_template(
        'results.html',
        recommendation=recommendation,
        statistics=statistics,
        plots=plots,
        peers=peers_df,
        sankey_html=sankey_html,
        countries=sorted(global_df['country'].dropna().unique()) if not global_df.empty else [],
        industries=sorted(global_df['industry'].dropna().unique()) if not global_df.empty else [],
        roles=sorted(global_df['role'].dropna().unique()) if not global_df.empty else [],
        methods=sorted(global_df['project_method'].dropna().unique()) if not global_df.empty else [],
        partners=sorted(global_df['external_partners'].dropna().unique()) if not global_df.empty else [],
        distributions=sorted(global_df['project_distribution'].dropna().unique()) if not global_df.empty else [],
        filter_country=request.args.get('filter_country'),
        filter_industry=request.args.get('filter_industry'),
        filter_role=request.args.get('filter_role'),
        filter_method=request.args.get('filter_method'),
        filter_partners=request.args.get('filter_partners'),
        filter_distribution=request.args.get('filter_distribution')
    )

if __name__ == '__main__':
    app.run(debug=True)
