There were 2 aims for my first NLP project.
1) JD insights for hiring managers on how to draft their JDs. 
2) Resume writing tip for job applicants.
The first area is a review of the Data roles in US using a kaggle data set https://www.kaggle.com/sl6149/data-scientist-job-market-in-the-us. 
EDA shows the major employers to be investment banks, consulting companies, specialised recruiting agencies and the big 4 Tech co.
NLP Topic Modelling with LSA and NMF using CountVectorizer and also TF-IDF was applied to look at the topics within the JDs after filtering for "Data Scien" in the job titles.
In the preliminary run, a particular company was found to be dominating one of the topics with the prevalence of its name. To avoid its impact overshadowing the entire study, JD from the company was reduced, bring the data set to ~ 1200.
The 4 conclusive topics derived were i) Technical skills, ii) Employers' envirionment/workplace support, iii) Goals that the company wants to achieve with the Data Scientist and iv) Stakeholders working with the Data Scientist. 

The second part is conducted using a regular JD and 3 genuine resumes. 
The focus is on the technical skillset match. Comparision is made on the first 80 words in the Skills ssection of each document. 
The comparison was applied using TF-IDF vectorizer and cosine similarity method.
The conclusion was as resume matching algorithms are increasingly used in the market, it would be to the advantage of job applicants to customize their resumes to the requirements of the JD.