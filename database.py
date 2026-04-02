import sqlite3

conn = sqlite3.connect('resume.db')

c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS candidates
(name TEXT, score REAL, skills TEXT)
''')

conn.commit()