import app
print('load_property_data...')
df = app.load_property_data()
print('loaded:', df.shape)
clean = app.prepare_listing_data(df)
print('prepared:', clean.shape)
print('columns:', list(clean.columns)[:30])
print('done')
