import graphlab
sf = graphlab.SFrame('/Users/hzdy1994/Downloads/people-example.csv')
#print sf
#print sf.tail() 
sf.show()
graphlab.canvas.set_target('ipynb')
sf['age'].show(view='Categorical')
#print sf['Country']
#print sf['age']
#print sf['age'].mean()
#print sf['age'].max()

#sf['Full Name'] = sf['First Name'] + ' ' + sf['Last Name']
#print sf

'''
def transform_country(country):
    if country == 'USA':
        return 'United States'
    else:
        return country
#print transform_country('Brazil')
#print transform_country('USA')
sf['Country'] = sf['Country'].apply(transform_country)
print sf
'''
