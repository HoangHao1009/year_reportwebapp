import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
import gdown

st.set_page_config(
    page_title = 'Báo Cáo Năm DELI Miền Nam 2023',
    page_icon = '📜',
    layout = 'wide'
)

options = option_menu(
    menu_title = 'Chọn chế độ',
    options = ['Chi tiết', 'Tổng quan'],
    orientation = 'horizontal'
)


@st.cache_data(persist = True)
def get_data(url):
    file_id = url.split("/")[-2]
    url = f"https://drive.google.com/uc?id={file_id}"
    data = pd.read_csv(url)
    success_empty = st.empty()
    success_empty.success("Đọc file CSV thành công. Dữ liệu DataFrame:")
    success_empty.empty()
    #data = pd.read_csv('donhang.csv')
    data = data[data['Kênh lớn'] == 'TRUYENTHONG']
    data = data[data['Nhóm lớn'] != '0']
    data = data[~data['Nhóm lớn'].isna()]
    data = data[data['Năm'] != 2021]
    data = data.drop(['Kênh lớn', 'Kênh nhỏ', 'Mã hàng cũ'], axis = 1)
    return data
url_empty = st.empty()
url = url_empty.text_input('Nhập url để xem báo cáo')
data = get_data(url)
url_empty.empty()

options_nhomlon = list(data['Nhóm lớn'].unique())

options_nhomlon = sorted([str(x) for x in options_nhomlon])
nhomlon = st.sidebar.multiselect(
    'Chọn nhóm lớn',
    options = ['Chọn tất cả'] + options_nhomlon,
    default = 'Chọn tất cả'
)

option_phanloai = sorted(data[data['Nhóm lớn'].isin(nhomlon)]['Phân loại'].unique().tolist())
phanloai = st.sidebar.multiselect(
    'Chọn phân loại',
    options = ['Chọn tất cả'] + option_phanloai,
    default = 'Chọn tất cả'
)

def set_all(nhom, col_name):
    if 'Chọn tất cả' in nhom or nhomlon == []:
        results = data[col_name].unique()
    else:
        results = nhom
    return results

nhomlon = set_all(nhomlon, 'Nhóm lớn')
phanloai = set_all(phanloai, 'Phân loại')


data = data[data['Nhóm lớn'].isin(nhomlon)
            & data['Phân loại'].isin(phanloai)]
@st.cache_data
def take_bc(group_col, data, type = 'normal'):
    def tangtruong(x, y, perc = False):
        df1 = x.copy()
        df2 = y.copy()
        def take_col(df):
            results = []
            for col in df.columns:
                if len(col) > 1:
                    x = col[-1]
                else:
                    x = col
                results.append(x)
            return results
        df1.columns = take_col(df1)
        df2.columns = take_col(df2)
        common_columns = df1.columns.intersection(df2.columns)

        if perc == False:
            output = df1[common_columns] - df2[common_columns]
        else:
            output = df1[common_columns] / df2[common_columns] - 1
            output = output.map(lambda x: f'{x:.2%}')
        return output
    group_col = group_col
    dt = data.groupby(group_col)['Doanh thu thuần'].sum()
    dt = dt.unstack(['Vùng miền', 'Năm', 'Khu vực'])

    dt_mb = dt.loc[:,['Miền Bắc']]
    dt_mn = dt.loc[:,['Miền Nam']]
    dt_2022 = dt.loc[:, (slice(None), 2022, slice(None))]
    dt_2023 = dt.loc[:, (slice(None), 2023, slice(None))]
    dt_mn_2022 = dt.loc[:,('Miền Nam', 2022, slice(None))]
    dt_mn_2023 = dt.loc[:,('Miền Nam', 2023, slice(None))]

    tangtruong_df = tangtruong(dt_mn_2023, dt_mn_2022, perc = True)
    tangtruong_df.fillna(0, inplace = True)
    tangtruong_df.columns = [f'{col}_tangtruong%' for col in tangtruong_df.columns]

    tangtruong_ds = tangtruong(dt_mn_2023, dt_mn_2022, perc = False)
    tangtruong_ds.fillna(0, inplace = True)
    tangtruong_ds.columns = [f'{col}_tangtruong' for col in tangtruong_ds.columns]

    sum_by_vm_2023 = dt_2023.groupby(level = 0, axis = 1).sum()
    sum_by_vm_2023.columns = [f'Tổng DS 2023 {col}' for col in sum_by_vm_2023]
    sum_by_vm_2023['Tổng DS 2023 Cả nước'] = sum_by_vm_2023.sum(axis = 1)
    sum_by_vm_2022 = dt_2022.groupby(level = 0, axis = 1).sum()
    sum_by_vm_2022.columns = [f'Tổng DS 2022 {col}' for col in sum_by_vm_2022]
    sum_by_vm_2022['Tổng DS 2022 Cả nước'] = sum_by_vm_2022.sum(axis = 1)
    sum_by_vm = pd.merge(sum_by_vm_2023, sum_by_vm_2022, left_index = True, right_index = True)
    sum_by_vm['Tăng trưởng MN 2023 (Phần trăm)'] = sum_by_vm['Tổng DS 2023 Miền Nam'] / sum_by_vm['Tổng DS 2022 Miền Nam'] -1
    sum_by_vm['Tăng trưởng MN 2023 (Doanh số)'] = sum_by_vm['Tổng DS 2023 Miền Nam'] - sum_by_vm['Tổng DS 2022 Miền Nam'] 
    sum_by_vm['Tỷ trọng MN so với Cả nước 2023'] = sum_by_vm['Tổng DS 2023 Miền Nam'] / sum_by_vm_2023['Tổng DS 2023 Cả nước']
    for i in ['Tăng trưởng MN 2023 (Phần trăm)', 'Tỷ trọng MN so với Cả nước 2023']:
        sum_by_vm[i] = sum_by_vm[i].map(lambda x: f'{x:.2%}')

    #DS 2022
    ds2022 = dt_mn_2022.copy()
    ds2022.columns = [f'Doanh số {x[-1]} {x[-2]}' for x in ds2022.columns]
    #DS 2023
    ds2023 = dt_mn_2023.copy()
    ds2023.columns = [f'Doanh số {x[-1]} {x[-2]}' for x in ds2023.columns]

    bc = pd.merge(sum_by_vm, tangtruong_df, how ='left', left_index = True, right_index = True)\
        .merge(tangtruong_ds, how = 'left', left_index = True, right_index = True)\
        .merge(ds2022, how = 'left', left_index = True, right_index = True)\
        .merge(ds2023, how = 'left', left_index = True, right_index = True)
    
    if type == 'normal':
        bc = bc
    elif type == 'nhomlon':
        bc['Tỷ trọng nhóm'] = bc['Tổng DS 2023 Miền Nam'] / sum(bc['Tổng DS 2023 Miền Nam'])
        bc['Tỷ trọng nhóm'] = bc['Tỷ trọng nhóm'].map(lambda x: f'{np.round(x, 2):.2%}')
    elif type == 'phanloai':
        def tytrongnhom(df):
            df = df.reset_index()
            results = []
            for k, v in zip(df['Nhóm lớn'], df['Phân loại']):
                dt = df[df['Nhóm lớn'] == k]
                tongnhom = sum(dt['Tổng DS 2023 Miền Nam'])
                dsphanloai = dt[dt['Phân loại'] == v]['Tổng DS 2023 Miền Nam'].values[0]
                tytrong = dsphanloai/tongnhom
                results.append(tytrong)
            return results
        bc['Tỷ trọng nhóm'] = tytrongnhom(bc)
        bc['Tỷ trọng nhóm'] = bc['Tỷ trọng nhóm'].map(lambda x: f'{np.round(x, 2):.2%}')
    return bc

if options == 'Chi tiết':

    bc_nl = take_bc(['Nhóm lớn', 'Vùng miền', 'Năm', 'Khu vực'], data, type = 'nhomlon')

    bc_pl = take_bc(['Nhóm lớn', 'Phân loại','Vùng miền', 'Năm', 'Khu vực'], data, type = 'phanloai')

    bc_sp = take_bc(['Nhóm lớn', 'Phân loại', 'Mã hàng','Vùng miền', 'Năm', 'Khu vực'], data)

    
    default_col = ['Tăng trưởng MN 2023 (Phần trăm)', 'Tổng DS 2023 Miền Nam',
                'Tỷ trọng MN so với Cả nước 2023',
                'Tăng trưởng MN 2023 (Doanh số)',
                'Tổng DS 2023 Miền Bắc', 'Tổng DS 2023 Cả nước',
                'Tổng DS 2022 Miền Nam', 'Tổng DS 2022 Miền Bắc', 'Tổng DS 2022 Cả nước',]
    mn_col = ['Tăng trưởng MN 2023 (Phần trăm)', 'Tỷ trọng MN so với Cả nước 2023',
            'Tổng DS 2023 Miền Nam', 'Tổng DS 2022 Miền Nam',
            'Tăng trưởng MN 2023 (Doanh số)',
            ]

    with st.expander('Click để chọn cột'):
        tatcamien = st.checkbox('Chọn cột của cả 2 miền (Mặc định miền Nam)')
        tatcacot = st.checkbox('Xem tất cả cột (ds chi tiết)')

    if tatcacot:
        if tatcamien:
            bc_nl = bc_nl
            bc_pl = bc_pl
            bc_sp = bc_sp
        else:
            another_col = [c for c in bc_sp if c not in default_col]
            col = another_col + mn_col
            bc_nl = bc_nl[col + ['Tỷ trọng nhóm']]
            bc_pl = bc_pl[col + ['Tỷ trọng nhóm']]
            bc_pl = bc_pl[col]
    else:
        if tatcamien:
            bc_nl = bc_nl[default_col + ['Tỷ trọng nhóm']]
            bc_pl = bc_pl[default_col + ['Tỷ trọng nhóm']]
            bc_sp = bc_sp[default_col]
        else:
            bc_nl = bc_nl[mn_col + ['Tỷ trọng nhóm']]
            bc_pl = bc_pl[mn_col + ['Tỷ trọng nhóm']] 
            bc_sp = bc_sp[mn_col]

    theonl, theopl = st.columns(2)
    with theonl:
        st.info('Báo cáo tổng hợp theo Nhóm lớn')
        st.data_editor(bc_nl.reset_index())
    with theopl:
        st.info('Báo cáo tổng hợp theo Phân loại')
        st.data_editor(bc_pl.reset_index())
    st.info('Báo cáo tổng hợp theo Sản phẩm')
    st.data_editor(bc_sp.reset_index())

elif options == 'Tổng quan':
    dt = data.copy()

    bc_nl = take_bc(['Nhóm lớn', 'Vùng miền', 'Năm', 'Khu vực'], data, 'nhomlon')

    bc_pl = take_bc(['Nhóm lớn', 'Phân loại','Vùng miền', 'Năm', 'Khu vực'], data, 'phanloai')

    bc_sp = take_bc(['Nhóm lớn', 'Phân loại', 'Mã hàng','Vùng miền', 'Năm', 'Khu vực'], data)

    with st.expander('FILTER MORE'):
        nam = st.selectbox('Chọn năm', options = [2023, 2022])
        chonhcm = st.checkbox('Lấy khu vực HCM')

    if chonhcm == True:
        khuvucmn = dt[dt['Vùng miền'] == 'Miền Nam']['Khu vực'].unique()
    else:
        khuvucmn = dt[dt['Vùng miền'] == 'Miền Nam']['Khu vực'].unique()
        khuvucmn = [i for i in khuvucmn if i != 'HCM']

    khuvucmn = sorted(khuvucmn)

    dt = dt[dt['Năm'] == nam]

    xemmienbac = st.checkbox('Xem tổng Miền Bắc')

    dstongvm = dt.groupby(['Vùng miền'])['Doanh thu thuần'].sum().reset_index()
    dstongmn = dstongvm.loc[dstongvm['Vùng miền'] == 'Miền Nam', :]['Doanh thu thuần']
    dstongmb = dstongvm.loc[dstongvm['Vùng miền'] == 'Miền Bắc', :]['Doanh thu thuần']

    if xemmienbac == False:
        st.info(f'Tổng doanh số miền Nam {nam}')
        st.metric('Doanh số', f'{dstongmn.values[0]:,}')
    else:
        mn, mb = st.columns(2)
        with mn:
            ds = np.round(dstongmn.values[0] / 1000000000, 2)
            st.info(f'Tổng doanh số miền Nam {nam}')
            st.metric('Doanh số', f'{ds} tỷ')
        with mb:
            ds = np.round(dstongmb.values[0] / 1000000000, 2)
            st.info(f'Tổng doanh số miền Bắc {nam}')
            st.metric('Doanh số', f'{ds} tỷ')
    
    dt_mn = dt[dt['Khu vực'].isin(khuvucmn)]

    ds_kv_nam_mn = dt_mn.groupby(['Khu vực'])['Doanh thu thuần'].sum().reset_index()


    kv_col = st.columns(len(khuvucmn))

    for i, k in enumerate(kv_col):
        with k:
            ds = ds_kv_nam_mn[ds_kv_nam_mn['Khu vực'] == khuvucmn[i]]['Doanh thu thuần'].values[0]
            ds = np.round(ds / 1000000000, 2)
            st.info(f'Tổng doanh số {khuvucmn[i]}')
            st.metric('Doanh số', f'{ds} tỷ')

    q1, q2 = st.columns(2)
    with q1:
        dsmien_px = px.pie(
            ds_kv_nam_mn,
            names = 'Khu vực',
            values = 'Doanh thu thuần',
            title = f'Tỷ trọng doanh số các khu vực'
        )
        st.plotly_chart(dsmien_px, use_container_width = True)
    with q2:
        if len(nhomlon) > 10:
            plotdata = bc_sp.groupby(['Nhóm lớn'])[f'Tổng DS {nam} Miền Nam'].sum().reset_index()
            plotdata_px = px.pie(
                plotdata,
                names = 'Nhóm lớn',
                values = f'Tổng DS {nam} Miền Nam',
                title = f'Tỷ trọng doanh số các nhóm'
            )
        else:
            plotdata = bc_sp.groupby(['Phân loại'])[f'Tổng DS {nam} Miền Nam'].sum().reset_index()
            plotdata_px = px.pie(
                plotdata,
                names = 'Phân loại',
                values = f'Tổng DS {nam} Miền Nam',
                title = f'Tỷ trọng doanh số phân loại'
            )
        st.plotly_chart(plotdata_px, use_container_width = True)


    p1, p2 = st.columns(2)
    with p1:
        if len(nhomlon) > 10:
            x = dt.groupby(['Nhóm lớn'])['Doanh thu thuần'].sum().reset_index()
            x = x.sort_values(by = ['Doanh thu thuần'])
            x_px = px.bar(
                x,
                x = 'Doanh thu thuần',
                y = 'Nhóm lớn',
                orientation = 'h',
                title = 'Doanh số nhóm',
                text_auto = True
            )
            st.plotly_chart(x_px, use_container_width = True)
        else:
            x = dt.groupby(['Phân loại'])['Doanh thu thuần'].sum().reset_index()
            x = x.sort_values(by = ['Doanh thu thuần'])
            x_px = px.bar(
                x,
                x = 'Doanh thu thuần',
                y = 'Phân loại',
                orientation = 'h',
                title = 'Doanh số nhóm',
                text_auto = True
            )
            st.plotly_chart(x_px, use_container_width = True)
    with p2:
        if len(nhomlon) > 10:
            tangtruong2023 = bc_nl.reset_index().copy()
            tangtruong2023['Tăng trưởng MN 2023 (Phần trăm)'] = tangtruong2023['Tăng trưởng MN 2023 (Phần trăm)'].map(lambda x: float(x.replace('%', '')))
            tangtruong2023 = tangtruong2023.sort_values(by = ['Tăng trưởng MN 2023 (Phần trăm)'], ascending = False)
            tangtruong2023_px = px.bar(
                tangtruong2023,
                x = 'Tăng trưởng MN 2023 (Phần trăm)',
                y = 'Nhóm lớn',
                orientation = 'h',
                title = 'Tăng trưởng nhóm',
                text_auto = True
            )
            st.plotly_chart(tangtruong2023_px, use_container_width = True)
        else:
            tangtruong2023 = bc_pl.reset_index().copy()
            tangtruong2023['Tăng trưởng MN 2023 (Phần trăm)'] = tangtruong2023['Tăng trưởng MN 2023 (Phần trăm)'].map(lambda x: float(x.replace('%', '')))
            tangtruong2023 = tangtruong2023.sort_values(by = ['Tăng trưởng MN 2023 (Phần trăm)'], ascending = False)
            tangtruong2023_px = px.bar(
                tangtruong2023,
                x = 'Tăng trưởng MN 2023 (Phần trăm)',
                y = 'Phân loại',
                orientation = 'h',
                title = 'Tăng trưởng nhóm',
                text_auto = True
            )
            st.plotly_chart(tangtruong2023_px, use_container_width = True)

    spbest, spworst = st.columns(2)
    with spbest:
        spb = dt.groupby(['Mã hàng'])['Doanh thu thuần'].sum().reset_index()
        spb = spb.sort_values(by = ['Doanh thu thuần'], ascending = False)
        spb['Mã hàng'] = spb['Mã hàng'].map(lambda x: str(x))
        spb = spb.iloc[:5, :]
        spb_px = px.bar(
            spb,
            x = 'Doanh thu thuần',
            y = 'Mã hàng',
            title = '5 sản phẩm chạy nhất',
            text_auto = True
        )
        st.plotly_chart(spb_px, use_container_width = True)
    with spworst:
        spb_w = dt.groupby(['Mã hàng'])['Doanh thu thuần'].sum().reset_index()
        spb_w = spb_w.sort_values(by = ['Doanh thu thuần'], ascending = True)
        spb_w['Mã hàng'] = spb_w['Mã hàng'].map(lambda x: str(x))
        spb_w = spb_w.iloc[:5, :]
        spb_w_px = px.bar(
            spb_w,
            x = 'Doanh thu thuần',
            y = 'Mã hàng',
            title = '5 sản phẩm kém nhất',
            text_auto = True
        )
        st.plotly_chart(spb_w_px, use_container_width = True)


