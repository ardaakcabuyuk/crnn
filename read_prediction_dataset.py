def read_dataset(api, path, filename):
    form_ids = []
    out = open(path + 'trimmed_' + filename, 'w+')
    writer = csv.writer(out)
    with open(path + filename, 'r') as f:
        content = f.readlines()[1:1000]
        for i, line in enumerate(tqdm(content)):
            form_id = line[:-1]
            try:
                form = api.get_form(form_id)
                writer.writerow([form_id])
                form_ids.append(form_id)
            except:
                print('form ' + form_id + ' does not exist')
    out.close()
    return form_ids

def get_form_urls(api, form_ids):
    urls = {}
    image_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    for form_id in tqdm(form_ids[127:]):
        urls[form_id] = []
        for sub in api.get_form_files(form_id):
            for ext in image_extensions:
                if sub['name'].endswith(ext):
                    if urllib.urlopen(sub['url'].encode('utf-8')).getcode() == 404:
                        if urllib.urlopen(('http://www.jotform.com/uploads/' + sub['username'] + '/form_files/' + sub['name']).encode('utf-8')).getcode() != 404:
                            urls[form_id].append('http://www.jotform.com/uploads/' + sub['username'] + '/form_files/' + sub['name'])
                    else:
                        urls[form_id].append(sub['url'])
                    break
    return urls

def create_url_dataset(path, filename, urls_dict):
    urls_file = open(path + 'urls_' + filename, 'w+')
    writer = csv.writer(urls_file, delimiter='|')
    for key in tqdm(urls_dict.keys()):
        data = [key]
        for url in urls_dict[key]:
            data.append(url.encode('utf-8'))
        if len(data) != 1:
            writer.writerow(data)

def create_image_dataset(path, url_dataset):
    with open(path + url_dataset) as f:
        for line in tqdm(f.readlines()[113:]):
            data = line[:-1].split('|')

            form_id = data[0]
            dir_name = path + 'images/' + form_id + '/'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            for img_no, url in enumerate(data[1:]):
                urllib.urlretrieve(url, dir_name + form_id + '_' + str(img_no + 1).encode('utf-8') + '.jpg')

def main():
    api_key = '-'
    path = 'Desktop/jotform/ocr_tool/'
    filename = 'submissionInJune408K.csv'
    url_dataset = 'urls_' + filename
    api = JotformAPIClient(api_key)
    form_ids = read_dataset(api, path, filename)
    print('Dataset read.')
    urls_by_form = get_form_urls(api, form_ids)
    print('Form URLs parsed.')
    create_url_dataset(path, filename, urls_by_form)
    print('URL dataset created.')
    create_image_dataset(path, url_dataset)
    print('Image dataset created.')
    print('Terminate.')

if __name__ == "__main__":
    main()
