from flask import Flask, render_template, request, session, redirect, url_for
import plotly.graph_objects as go
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import secrets
import os

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

matplotlib.use('Agg')

# === Files ===
CSV_FILE = 'user_responses_.csv'
CAUSAL_EFFECTS_CSV = 'model_outputs/causal_effects.csv'

# Load causal effects once at startup
causal_df = pd.read_csv(CAUSAL_EFFECTS_CSV, encoding='utf-8-sig') if os.path.exists(CAUSAL_EFFECTS_CSV) else pd.DataFrame()
print("Loaded causal_effects.csv rows:", len(causal_df) if not causal_df.empty else 0)


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


def get_collaborative_recommendation(df, current_user_response):
    user_continent = COUNTRY_TO_CONTINENT.get(current_user_response['country'], None)
    if not user_continent:
        return [f"No continent mapping available for '{current_user_response['country']}'."], pd.DataFrame()

    team_size = current_user_response['team_size']
    min_size, max_size = team_size * 0.75, team_size * 1.25

    df = df.copy()
    df['continent'] = df['country'].map(COUNTRY_TO_CONTINENT)

    df = df[
    ~(
        (df['country'] == current_user_response['country']) &
        (df['industry'] == current_user_response['industry']) &
        (df['team_size'] == current_user_response['team_size']) &
        (df['team_customer_relationship'] == current_user_response['team_customer_relationship']) &
        (df['team_communication'] == current_user_response['team_communication'])
    )
]
    peers = df[
        (df['continent'] == user_continent) &
        (df['industry'] == current_user_response['industry']) &
        (df['team_size'].astype(float).between(min_size, max_size)) 
    ]
    print('peers are here', peers)
    if peers.empty:
        return [f"No peer companies available in {user_continent}, {current_user_response['industry']} with similar size."], pd.DataFrame()

    avg_customer_rel = peers['team_customer_relationship'].astype(float).mean()
    avg_team_comm = peers['team_communication'].astype(float).mean()

    advice = []
    if current_user_response['team_customer_relationship'] <= avg_customer_rel:
        advice.append(f"Your team-customer relationship ({current_user_response['team_customer_relationship']}) is below peers ({avg_customer_rel:.2f}).")
    elif current_user_response['team_customer_relationship'] > avg_customer_rel:
        advice.append(f"Your team-customer relationship ({current_user_response['team_customer_relationship']}) is above peers ({avg_customer_rel:.2f}).")

    if current_user_response['team_communication'] <= avg_team_comm:
        advice.append(f"Your internal communication ({current_user_response['team_communication']}) is below peers ({avg_team_comm:.2f}).")
    elif current_user_response['team_communication'] >= avg_team_comm:
        advice.append(f"Your internal communication ({current_user_response['team_communication']}) is above peers ({avg_team_comm:.2f}).")

    return advice, peers

def get_causal_recommendation(user_response, causal_df):
    if causal_df.empty:
        return None, None   # return both text recs and raw data

    problems = [user_response.get('top1'), user_response.get('top2'), user_response.get('top3')]
    recs = []
    edges = []  # 🟢 for Sankey diagram

    for problem in problems:
        if not problem:
            continue

        matches = causal_df[causal_df['Problem'] == problem]
        if matches.empty:
            recs.append(f"No causal recommendations available for '{problem}'.")
            continue

        for _, row in matches.iterrows():
            treatment = row['Treatment']
            effect = row['EffectSize']
            percentage = abs(effect) * 100

            if effect < 0:
                recs.append(f"Improving {treatment} is likely to DECREASE the '{problem}' problem (strength {percentage:.1f}%).")
            else:
                recs.append(f"Improving {treatment} is likely to INCREASE the '{problem}' problem (strength {percentage:.1f}%).")

            # 🟢 store structured edge for Sankey
            edges.append((problem, treatment, effect))

    return recs, edges



# === Routes ===
@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    df = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame()
    if 'team_size' in df.columns:
        df['team_size'] = pd.to_numeric(df['team_size'], errors='coerce').astype('Int64')
    industries = sorted(df['industry'].unique()) if not df.empty else []
    countries = sorted(df['country'].unique()) if not df.empty else []

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
            'project_method': request.form['project_method'],
            'meeting_frequency': request.form['meeting_frequency'],
            'user_feedback_frequency': request.form['user_feedback_frequency'],
            'assumption': request.form['assumption'],
            'common_problem': request.form['common_problem'],
            'top2': request.form['top2'],
            'top3': request.form['top3'],
            'top4': request.form['top4'],
            'top5': request.form['top5']
        }

        session['user_response'] = response
        df = pd.concat([df, pd.DataFrame([response])], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)

        return redirect(url_for('results'))

    return render_template('survey.html', countries=countries, industries=industries)


@app.route('/results')
def results():
    df = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame()
    user_response = session.get('user_response')
    if not user_response:
        return redirect(url_for('survey'))

    filter_country = request.args.get('filter_country')
    filter_industry = request.args.get('filter_industry')
    filter_role = request.args.get('filter_role')
    filter_method = request.args.get('filter_method')
    filter_partners = request.args.get('filter_partners')
    filter_distribution = request.args.get('filter_distribution')

    filtered_df = df.copy()
    if filter_country:
        filtered_df = filtered_df[filtered_df['country'] == filter_country]
    if filter_industry:
        filtered_df = filtered_df[filtered_df['industry'] == filter_industry]

    if filter_role:
        filtered_df = filtered_df[filtered_df['role'] == filter_role]
    if filter_method:
        filtered_df = filtered_df[filtered_df['project_method'] == filter_method]
    if filter_partners:
        filtered_df = filtered_df[filtered_df['external_partners'] == filter_partners]
    if filter_distribution:
        filtered_df = filtered_df[filtered_df['project_distribution'] == filter_distribution]


    causal_recs, causal_edges = get_causal_recommendation(user_response, causal_df)
    collab_recs, peers_df = get_collaborative_recommendation(df, user_response)

    recommendation = causal_recs if causal_recs else []
    statistics = collab_recs if collab_recs else []

        # === Sankey Diagram ===
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


        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                align="left",    
                color="lightblue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors
            )
        )])

        fig.update_layout(
    title_text="Causal Recommender Alluvial Diagram",
    font=dict(size=12, color="black"),
    margin=dict(r=200, t=50, b=50, l=50),  # extra right margin
    annotations=[
        dict(
            x=1.5, y=1.05,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            text="<b>Legend</b>",
            font=dict(size=12, color="black")
        ),
        dict(
            x=1.5, y=1.0,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            text="<span style='color:lightgreen;'>■</span> Positive relation",
            font=dict(size=11, color="black")
        ),
        dict(
            x=1.5, y=0.95,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            text="<span style='color:lightcoral;'>■</span> Negative relation",
            font=dict(size=11, color="black")
        )
    ]
)


        sankey_html = fig.to_html(full_html=False)


    plots = {}
    plot_dir = 'static'
    for column in [
        'team_customer_relationship', 'failed_documentation', 'team_communication', 'role',
        'industry',  'country', 'team_size', 'external_partners',
        'project_distribution', 'work', 'project_method', 'meeting_frequency',
        'user_feedback_frequency','common_problem', 'top2','top3','top4', 'top5'
    ]:
        plt.figure(figsize=(10, 6))
        if pd.api.types.is_numeric_dtype(filtered_df[column]):
            values = filtered_df[column].value_counts().sort_index()
        else:
            values = filtered_df[column].value_counts()

        bars = values.index.tolist()
        heights = values.values.tolist()
        colors = ['yellow' if str(bar) == str(user_response[column]) else 'blue' for bar in bars]

        plt.bar(bars, heights, color=colors)
        plt.title(column.replace('_', ' ').title())
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        path = os.path.join(plot_dir, f'{column}.png')
        plt.savefig(path)
        plots[column] = f'{column}.png'
        plt.close()

    return render_template(
    'results.html',
    recommendation=recommendation,
    statistics=statistics,
    plots=plots,
    peers=peers_df,
    sankey_html=sankey_html,   # 🟢 pass diagram
    countries=sorted(df['country'].unique()) if not df.empty else [],
    industries=sorted(df['industry'].unique()) if not df.empty else [],
    roles=sorted(df['role'].dropna().unique()) if not df.empty else [],
    methods=sorted(df['project_method'].dropna().unique()) if not df.empty else [],
    partners=sorted(df['external_partners'].dropna().unique()) if not df.empty else [],
    distributions=sorted(df['project_distribution'].dropna().unique()) if not df.empty else [],
    filter_country=filter_country,
    filter_industry=filter_industry,
    filter_role=filter_role,
    filter_method=filter_method,
    filter_partners=filter_partners,
    filter_distribution=filter_distribution
)



if __name__ == '__main__':
    app.run(debug=True)
