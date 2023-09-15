
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih


#########################
# GÖREV 1: Veriyi Hazırlama
#########################
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_csv("/Users/ardaugurlu/Documents/miuul/recommender_systems/ArmutARL-221114-234936/armut_data.csv")
df = df_.copy()




# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

#Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

#Tarih bilgisini içeren yeni bir 'date' sütunu oluşturma

df['New_Date'] = pd.to_datetime(df['CreateDate'], format='%Y-%m-%d %H:%M:%S').dt.to_period('M')

df.head()

# UserID ve date sütunlarını birleştirerek yeni bir 'ID' sütunu oluşturma

df['SepetID'] = df['UserId'].astype(str) + '_' + df['New_Date'].astype(str)

df.head()



#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

sepet_hizmet_pivot = pd.pivot_table(df, values='ServiceId', index='SepetID', columns='hizmet',
                                    aggfunc=lambda x: 1 if len(x) > 0 else 0, fill_value=0)


sepet_hizmet_pivot = sepet_hizmet_pivot.astype(bool)


#Adım 2:  Birliktelik kurallarını oluşturunuz
# Pivot tabloyu oluşturduğunuzu varsayalım ve sepet_hizmet_pivot adıyla bir DataFrame'e sahip olduğunuzu düşünelim.

# Frequent itemsets oluşturma
frequent_itemsets = apriori(sepet_hizmet_pivot, min_support=0.01, use_colnames=True)

# Support değerine göre sıralama
frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

# Birliktelik kurallarını oluşturma
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

# Support, confidence ve lift değerlerine göre filtreleme

rules[(rules["support"]>0.1) & (rules["confidence"]>0.1) & (rules["lift"]>1)]. \
sort_values("lift", ascending=False)





#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules,"2_0",5)

