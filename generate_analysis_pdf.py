import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'DataFlix - Dataset & Feature Analysis', ln=True, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        # Handle unicode safely
        body = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 6, body)
        self.ln()

content = """
1. Dataset-by-Dataset Analysis
------------------------------
Netflix Kaggle Dataset
- Netflix_Dataset_Rating.csv: Users, Ratings, Movie_IDs. Ratings are 1-5, no timestamps.
- Netflix_Dataset_Movie.csv: Movie_ID, Year, Name. Lightweight metadata.
Key finding: Netflix rating file covers 1,350 unique movies out of 17,770.

MovieLens 25M
- ratings.csv: Dense interaction matrix, has timestamps for temporal splits.
- movies.csv: Movie identity + genres.
- links.csv: Bridge to TMDb/IMDb.
- tags.csv / genome-scores.csv: Optional but extremely large. Cost 500MB+ in RAM.

Local TMDb API/CSV
- Provides: overview, release date, original language, runtime, popularity.
- This is the source of text for SBERT embeddings.

2. Recommended Dataset Usage
----------------------------
- Netflix Rating CSV: YES (Validation + Testing only)
- Netflix Movie CSV: YES (Movie title/year lookup for alignment)
- MovieLens ratings.csv: YES (Primary training interaction matrix)
- MovieLens movies.csv: YES (Genre feature table)
- MovieLens links.csv: YES (Bridge to TMDb API)
- MovieLens genome-scores.csv / tags.csv: NO (Too large, subsumed by SBERT)
- TMDb: YES (Plot synopses for SBERT embeddings)
- IMDb: NO (Redundant)

3. Merge Strategy
-----------------
Step 1: Netflix Movie CSV <-> MovieLens movies.csv (Fuzzy Match title+year)
Step 2: MovieLens movies.csv <-> MovieLens links.csv (by movieId)
Step 3: movieId -> tmdb_id -> TMDb CSV overview
Step 4: Unified contiguous integer ID mapping for all data.

4. Leakage and Harmful Columns
------------------------------
- vote_average (TMDb): DIRECT LEAKAGE. Aggregate of all user ratings.
- vote_count (TMDb): Indirect Leakage. Correlates with rating distribution.
- Netflix Movie_ID replacing ML Movie_ID: WRONG KEY. Spaces are disjoint.

5. Required Preprocessing Steps
-------------------------------
1. Temporal split on MovieLens (80/10/10).
2. Random user-level split on Netflix (50/50).
3. Filter ML sparse interactions.
4. Fuzzy title alignment.
5. Drop rows with unmapped Netflix IDs.
6. Mean-center ratings per user for Matrix Factorization.
7. BPR binarisation (centered rating > 0 -> positive).
8. Genre one-hot embeddings.
9. SBERT on TMDb overviews.
10. Log-normalise popularity scalars.
11. Pop-proportional negative sampling for BPR.

6. Final Strict Decision
------------------------
- USE: Netflix (Val/Test), MovieLens (Train), TMDb (Embeddings).
- DO NOT USE: genome-scores, tags, IMDb, external APIs (use local TMDB CSV).
- MERGE BY: Fuzzy Title Match + Year -> MovieLens ID -> links.csv -> TMDb.
- REMOVE COLUMNS: vote_average, vote_count, tagline, spoken_languages.
- VERDICT: The data is highly sufficient for the DataFlix hybrid architecture!
"""

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Split into chapters
for section in content.split('\\n\\n'):
    if section.strip():
        lines = section.strip().split('\\n')
        title = lines[0]
        body = '\\n'.join(lines[1:])
        pdf.chapter_title(title)
        pdf.chapter_body(body)

pdf.output('DataFlix_Dataset_Analysis.pdf')
print('DataFlix_Dataset_Analysis.pdf generated successfully')
