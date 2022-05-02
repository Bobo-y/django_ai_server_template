def title_to_name_map(title_list):
    title_index_map = {t: i for i, t in enumerate(title_list)}
    return title_index_map


def categories_to_id_map(categories):
    categories_id_map = {i['label_id']: i['name'] for i in categories}
    return categories_id_map


def categories_to_name_map(categories):
    categories_id_map = {i['name']: i['label_id'] for i in categories}
    return categories_id_map