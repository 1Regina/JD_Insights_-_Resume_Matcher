#%%
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

#%% 
#Wordcloud of job_titles
%config InlineBackend.figure_formats = ['retina']     # sets backend to render higher res images
df_all = pd.read_csv(r"C:\Users\regina\Desktop\Metis\Metis Projects\JD Resume Matcher\US JD data.csv")
wordcloud = WordCloud(background_color='white',colormap = "viridis", max_font_size = 50).generate_from_frequencies(df_all['position'].value_counts())
# wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#%% #Wordcloud of companies
%config InlineBackend.figure_formats = ['retina']     # sets backend to render higher res images
df_data_scientist = pd.read_csv(r"C:\Users\regina\Desktop\Metis\Metis Projects\JD Resume Matcher\Data Scien.csv")
wordcloud = WordCloud(background_color='white',colormap = "inferno", max_font_size = 50).generate_from_frequencies(df_data_scientist['company'].value_counts())
# wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#%%
#Wordcloud of location
%config InlineBackend.figure_formats = ['retina']     # sets backend to render higher res images
wordcloud = WordCloud(background_color='white',colormap = "magma", max_font_size = 50).generate_from_frequencies(df_data_scientist['location'].value_counts())
# wordcloud.recolor(color_func = grey_color_func)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

plt.show()

#%%
