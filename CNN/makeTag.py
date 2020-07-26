import json
import pandas as pd
import numpy as np


test = pd.read_json('/root/data/test.json', typ='frame')
genre_gn_all = pd.read_json('/root/data/genre_gn_all.json', typ = 'series')
song_meta = pd.read_json('/root/data/song_meta.json', typ = 'frame')


genre_gn_all = pd.DataFrame(genre_gn_all, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})
gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] == '00']

dtl_gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] != '00']
dtl_gnr_code.rename(columns = {'gnr_code' : 'dtl_gnr_code', 'gnr_name' : 'dtl_gnr_name'}, inplace=True)

gnr_code = gnr_code.assign(join_code = gnr_code['gnr_code'].str[0:4])


song_gnr_map = song_meta.loc[:, ['id', 'song_gn_gnr_basket']]

# unnest song_gn_gnr_basket
song_gnr_map_unnest = np.dstack(
    (
        np.repeat(song_gnr_map.id.values, list(map(len, song_gnr_map.song_gn_gnr_basket))),
        np.concatenate(song_gnr_map.song_gn_gnr_basket.values)
    )
)

# unnested 데이터프레임 생성 : song_gnr_map
song_gnr_map = pd.DataFrame(data = song_gnr_map_unnest[0], columns = song_gnr_map.columns)
song_gnr_map['id'] = song_gnr_map['id'].astype(str)
song_gnr_map.rename(columns = {'id' : 'song_id', 'song_gn_gnr_basket' : 'gnr_code'}, inplace = True)

# unnest 객체 제거
del song_gnr_map_unnest



# 플레이리스트 아이디(id)와 수록곡(songs) 추출
plylst_song_map = test[['id', 'songs']]

# unnest songs
plylst_song_map_unnest = np.dstack(
    (
        np.repeat(plylst_song_map.id.values, list(map(len, plylst_song_map.songs))),
        np.concatenate(plylst_song_map.songs.values)
    )
)

# unnested 데이터프레임 생성 : plylst_song_map
plylst_song_map = pd.DataFrame(data = plylst_song_map_unnest[0], columns = plylst_song_map.columns)
plylst_song_map['id'] = plylst_song_map['id'].astype(int)
plylst_song_map['songs'] = plylst_song_map['songs'].astype(int)
plylst_song_map['id'] = plylst_song_map['id'].astype(str)
plylst_song_map['songs'] = plylst_song_map['songs'].astype(str)

# unnest 객체 제거
del plylst_song_map_unnest


# 플레이리스트 아이디(id)와 매핑된 태그(tags) 추출
plylst_tag_map = test[['id', 'tags']]

# unnest tags
plylst_tag_map_unnest = np.dstack(
    (
        np.repeat(plylst_tag_map.id.values, list(map(len, plylst_tag_map.tags))),
        np.concatenate(plylst_tag_map.tags.values)
    )
)

# unnested 데이터프레임 생성 : plylst_tag_map
plylst_tag_map = pd.DataFrame(data = plylst_tag_map_unnest[0], columns = plylst_tag_map.columns)
plylst_tag_map['id'] = plylst_tag_map['id'].astype(str)

# unnest 객체 제거
del plylst_tag_map_unnest

plylst_song_tag_map = pd.merge(plylst_song_map, plylst_tag_map, how = 'left', on = 'id')

# 2. 1번 테이블 plylst_song_tag_map + 곡 장르 테이블 song_gnr_map join
plylst_song_tag_map = pd.merge(plylst_song_tag_map, song_gnr_map, how = 'left', left_on = 'songs', right_on = 'song_id')

# 4. 최종 테이블 필드 선택
plylst_song_tag_map = plylst_song_tag_map[['id', 'songs', 'tags']]

plylst_song_tag_map = plylst_song_tag_map.dropna(axis=0)

plylst_song_tag_map = plylst_song_tag_map.groupby('songs')['tags'].apply(list).reset_index(name='new')


result = []
with open("./result_songs.json", "r") as fp:
    result = eval(fp.read())


for i in range(len(result)):
    tag_num = 0
    print(i)
    if len(result[i]['songs']) == 0 :
        result[i]['songs'] = result[i-1]['songs']
        result[i]['tags'] = result[i-1]['tags']

    else :
        result[i]['tags'] = set()
        for song in result[i]['songs']:
            tags = list(plylst_song_tag_map.loc[plylst_song_tag_map['songs'] == song]['new'])
            if tags:
                result[i]['tags'].update(tags[0][:10])

            if len(result[i]['tags']) >= 10:
                result[i]['tags'] = list(result[i]['tags'])[:10]
                break
        result[i]['tags'] = list(result[i]['tags'])


with open("./result.json" , "w") as fp :
    json.dump(result, fp)


