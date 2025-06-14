from bs4 import BeautifulSoup
import requests
import random
import time
import csv
import pandas as pd 
import numpy as np
import re
import os
import json
from lxml import etree
# url = 'https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments&client=pc&clientVersion=1.0.0&t=1727317244350&body=%7B%22productId%22%3A100066930471%2C%22score%22%3A0%2C%22sortType%22%3A5%2C%22page%22%3A5%2C%22pageSize%22%3A10%2C%22isShadowSku%22%3A0%2C%22rid%22%3A0%2C%22fold%22%3A1%2C%22bbtf%22%3A%22%22%2C%22shield%22%3A%22%22%7D&h5st=20240926102044365%3Brdd1skcx2obrbdb5%3Bfb5df%3Btk03wae611cb418n3iXy9HqIh9MjuiTkdRXuHo7JpygGCKY-6K2l1hG0AE1cemAtz_esdg5IaUcB6IuydbhftZxaEZdP%3B555be28f41cd1410ce419fe16a65af0dd91e8a303f3ee3ff36155f7d788ec65b%3B4.8%3B1727317244365%3BUOG3MOUIoSzLmOUOcOEKhNwO2W0UqOEJ0Fw_kNg7k2f_rJ-KyFw60W0I0Wv90WUOMmUI2uzOoJg_0WUOMm0OmSDIw5TIiGQJfOg_eOTKfWT91RD9fGj_k_w_dSD9jSwO2uzOm9S7vNA9fNUO2uWL0KzJfSA9kGzKdOTKfKw_w9w_lKT9xBD_hCA9x9AJmSzK0W0I0K-_gdA_zNUO2uWL0SzLeeUJgZB9qxw93BT80W0I0_vO2W0UqO0CdCg791B_TZA9BNUOcOE7nFw71NvO2W0UqO0KjeA8-NUOcO09mNUO2uWLZVUOMWTOcOUIhNwO2WUO2uWL0OUOcOkJhNwO2WUO2uWLgGTOcO0JhNwO2WUO2uWLmW0I0CD50NUO2WUOMmUK2uzOiCv_0WUO2W0UqyTI2uzOjCv_0WUO2W0UqSDL2uzOkCv_0WUO2W0UqOEIoSzLmOUOcOEKhNwO2WUO2uWLmW0I0qg50WUO2W0UqCTOcOk6qNUO2WUOMmUK2uzOqNUO2WUOMmUK2uzOy5vO2WUO2um42uzOydA9i1-90WUOMmE3bV0I0WP60WUOMm0Oi_T42qTJgeA8-VkImeUKlWUBIVk6fZQ9oxgB0W0I0SA5jNUO2um4%3Ba7ac90eaac55163ad8d601913a7f50de10f53ead7a6becc4af7b766d1018ad45&x-api-eid-token=jdd03LMLLLSS7AEJZXB2CSWJA7EBQMJHPS2QCAPCW3PJTI4BTKFVSQQ2O23QP4OUSKJAK2TIIZLYXT7MTTUN5FXHNLXHOUIAAAAMSFQMLQWIAAAAADUHI4JEBNVD67MX&loginType=3&uuid=181111935.1720761500341523571790.1720761500.1727262460.1727315135.5'
# headers = {
#     'user-agent' :'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0',
#     'cookie': 'b_dw=1167; b_dpr=1.5625; b_webp=1; b_avif=1; shshshfpa=c9b00b7b-ff0e-4b8f-66ef-8b488d1b0e8f-1720761502; shshshfpx=c9b00b7b-ff0e-4b8f-66ef-8b488d1b0e8f-1720761502; __jdu=1720761500341523571790; TrackID=1osTWrxCFziaVpxa8aqxUCWqdkmZH-yXpjYpZfdnIvmiLGXITq7tezQXiha2HHDRldeiEhEifexQ4ZGTqbGNuTjqlZslTBE8GFZreJJkvz8Wbq3dBUg_Zq-nsofb61pdz|||HkwNzvQjt0tHDdg1P7YwXw; pinId=HkwNzvQjt0tHDdg1P7YwXw; thor=DD2B29B104FAF457B771DF51CEF921C3A94684DAF6554D571EBC1010A5DC3B23B646A6C2A527BA29365987562592F1B8A6AD2819D1B5E2FC3BFAD542178D6107BEC9733A525F758559609F8D245C7714A23096683C0BD919937F6CBF490CE9B782707B8A3FB1B4DB55EE8286382533A2B8F5D854C37C4E5E9DBA7FFC07DA72926AA6D6D173CD4D8369E073912D395CDA; __jdv=181111935|direct|-|none|-|1727262459856; areaId=17; ipLoc-djd=17-1381-50712-62966; b_dh=560; autoOpenApp_downCloseDate_auto=1727263333070_1800000; cn=0; TARGET_UNIT=bjcenter; 3AB9D23F7A4B3C9B=LMLLLSS7AEJZXB2CSWJA7EBQMJHPS2QCAPCW3PJTI4BTKFVSQQ2O23QP4OUSKJAK2TIIZLYXT7MTTUN5FXHNLXHOUI; PCSYCityID=CN_420000_420100_0; token=3269ba41d2814af1c522ad9c6642c7d6,3,959619; jsavif=1; 3AB9D23F7A4B3CSS=jdd03LMLLLSS7AEJZXB2CSWJA7EBQMJHPS2QCAPCW3PJTI4BTKFVSQQ2O23QP4OUSKJAK2TIIZLYXT7MTTUN5FXHNLXHOUIAAAAMSFQE5C5AAAAAADGITKHTJJZI564X; _gia_d=1; __jda=181111935.1720761500341523571790.1720761500.1727262460.1727315135.5; __jdb=181111935.3.1720761500341523571790|5.1727315135; __jdc=181111935; flash=3_GjIIaqv93YzLOn-9XOoh3Tjfi3jAOJEUzgkcN3QCKGiLdncv41BYPrmyul_KmI6fAzfN1Qy5B0020SAbPpGdM_IePANKdjjNgFVZeVCS-9sBlvY4TYjGZVu7SJpJ09I5SGFQ6G7eyaLpgrkLiCBW3xsc8nfU1hSM69WkUWZcdxk*; shshshfpb=BApXScEwBL_dARqVrDUmlO9Qm4j823b9VBmI3Ei1p9xJ1MiGHjYC2'
# }


# #for i in [1727317748757,1727317832497,1727317887877,1727317914200,1727318213161,1727318445594,1727318547481]:
# param = {
#   'appid': 'item-v3',
#   'functionId': 'pc_club_productPageComments',
#   'client': 'pc',
#   'clientVersion': '1.0.0',
#   't': '1727317748757',
#   'body': '%7B%22productId%22%3A100066930471%2C%22score%22%3A0%2C%22sortType%22%3A5%2C%22page%22%3A2%2C%22pageSize%22%3A10%2C%22isShadowSku%22%3A0%2C%22rid%22%3A0%2C%22fold%22%3A1%2C%22bbtf%22%3A%22%22%2C%22shield%22%3A%22%22%7Dh5st: 20240926101336205%3Brdd1skcx2obrbdb5%3Bfb5df%3Btk03wae611cb418n3iXy9HqIh9MjuiTkdRXuHo7JpygGCKY-6K2l1hG0AE1cemAtz_esdg5IaUcB6IuydbhftZxaEZdP%3B6d6dde732f69066e93ee410c26cb273e29a51dc14ffcd4837a56100c19941442%3B4.8%3B1727316816205%3BUOG3MOUIoSzLmOUOcOEKhNwO2W0UqOEJ0Fw_kNg7k2f_rJ-KyFw60W0I0Wv90WUOMmUI2uzOoJg_0WUOMm0OmSDIw5TIiGQJfOg_eOTKfWT91RD9fGj_k_w_dSD9jSwO2uzOm9S7vNA9fNUO2uWL0KzJfSA9kGzKdOTKfKw_w9w_lKT9xBD_hCA9x9AJmSzK0W0I0K-_gdA_zNUO2uWL0SzLeeUJgZB9qxw93BT80W0I0_vO2W0UqOkGkNj5CFj88tR8uNUOcOE7nFw71NvO2W0UqO0KjeA8-NUOcO09mNUO2uWLZVUOMWTOcOUIhNwO2WUO2uWL0OUOcOkJhNwO2WUO2uWLgGTOcO0JhNwO2WUO2uWLmW0I0CD50NUO2WUOMmUK2uzOiCv_0WUO2W0UqWDI2uzOjCv_0WUO2W0UqSDL2uzOkCv_0WUO2W0UqOEIoSzLmOUOcOEKhNwO2WUO2uWLmW0I0qg50WUO2W0UqCTOcOk6qNUO2WUOMmUK2uzOqNUO2WUOMmUK2uzOy5vO2WUO2um42uzOydA9i1-90WUOMmE3bV0I0WP60WUOMm0Oi_T42qTJgeA8-VkImeUKlWUBIVk6fZQ9oxgB0W0I0SA5jNUO2um4%3B269c3dd1f97677eec9b4714ff8ca391701d21db75c41d8b0b2f6eb701f6b760a',
#   'x-api-eid-token': 'jdd03LMLLLSS7AEJZXB2CSWJA7EBQMJHPS2QCAPCW3PJTI4BTKFVSQQ2O23QP4OUSKJAK2TIIZLYXT7MTTUN5FXHNLXHOUIAAAAMSFQMLQWIAAAAADUHI4JEBNVD67MX',
#   'loginType':'3',
#   'uuid': '181111935.1720761500341523571790.1720761500.1727262460.1727315135.5'
# }

# response = requests.get(url, headers=headers,params=param)
# data = response.text

# #print(data)

# json_data = json.loads(data)

# comments = json_data.get("comments", [])

# content_list = [comment.get("content", "") for comment in comments]

# #print(content_list)

# df = pd.DataFrame(content_list)

# df.to_excel('comments.xlsx', index = True)

# #with open('comment.csv', 'w', encoding='utf-8') as file:
# #    writer = csv.writer(file)
# #    for content in content_list:
# #        writer.writerow([content])

# url = 'https://www.nba.com/stats/teams/traditional?Season=1996-97&SeasonType=Playoffs&dir=A&sort=W'
# headers ={
#   'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
#   'cookie':'bm_mi=FA5500E7F885A4E945F6C2AB89AC1782~YAAQl4zZFxlj1m+SAQAACDDylBkaCHKY1ltcgUPyYdnYLsiB+mgtR0gCRO/AbctaWH6IVgaddD+WVK5/vyUwry/vPVk/xjcQx1LsA3Vge+KeZnHj/wj2/4ZcfXDH+Oxmw0T+3PtObdaxo22G5OSSlWWswvgbr4wLBqw1JXkpzpmggClwZfWCE1QuFebHmnN7bwQUqEHqU+kmWC/Vx8NbBSWNPmsSXaX0/eWAWJp2qVKpmuV7ZFfUego4idL3RYpRmCan0r1AZj5Sbu8/SZSj+weenBrYNIS2vAVsadGktSfZcJiWAPU3sqjxZBqMBrY4IttYwA==~1; _abck=24952EC4555213DF1D63C5D7C6960075~0~YAAQn4zZF/d4TnqSAQAAB3z6lAyV1FkbNsl4pYQInFLhNl1LB6efdUgpoOO+wpwrdwHa3Y2pwlWuN7tDInkHYHb6Fsi+2I3GQZr1mU1aTMGfvWpOD5yTzHkl0jiV0ZieWimRLalTGOkddrglEiCq3tFF12jp54w6j0jEJk7chfs5kJqadQQWNxqjnxiFYhdUsQj/EaEwX8a7YwOO+DxDQ2hHLTwk5D55RRXcUI2MSe+rXbpXpitGtLV3LpWagj0HJL44o67KNcLvmfI6Mf2WdMf6We+YUphQdFTCyT/YidPYEQs1CMU6qxI10gr/vgv6VNkcr8svE54941FTFNUmHEfAwDVPtN6ISb50rZHLvob6a70GFjdmROWQmhJmedn9Q91taVUVwQSazPFXWh78ISg03Y6XnnVLPYxU80XojCfW2j2eXsUPtKEBDilCwsq/MN/4WvkL~-1~||0||~1729079896; usprivacy=1---; AMCVS_248F210755B762187F000101%40AdobeOrg=1; s_ecid=MCMID%7C63330122577559355533387036496701478061; AMCV_248F210755B762187F000101%40AdobeOrg=179643557%7CMCIDTS%7C20013%7CMCMID%7C63330122577559355533387036496701478061%7CMCAAMLH-1729681099%7C11%7CMCAAMB-1729681099%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1729083500s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C5.5.0; mediakindauth2token=AuthToken1fZBfb5swFMU_TXiZgsD_sB94QEmbRFu1Ssua7mky-JK4Acxs04R9-pmsWdWpm2RdX537O746LgaloasgP3jfuxkuZug2HAvNqLt9L60f494aJeOulLF_xlVjBhVXpg1UtHFuAPuX13n3H8vEvqLVAVrp4nPbOCP72Nh9UE8uFJQkNFyXohV0XvsxtFUjdTvNO9lCvh_A-e8Kajk0PtqaI3TbsYd8NenRFjrZ-Y3Kr8ASnnUFF2Jj3FaWDfioaBpzAnWF3R_6qwMbzG93FFVlhsujb_Xl7_tdz7uRW11Z40ztXz7mGppPodk_QlvTvIT-MIRNUTH4g7H6p_TadHfSHXOMeZoJniQZ5YwSkTLGKKNCIJ5hwadpihjHaZJyRAlKBSMk41TQlNKETBThhGaCcYoQExQzFOYJj27OvbbgPnd5muE0oyRLULSwID2oi4hEkjEcxI8wrgat8hIxzGuE5jUQPCeCo3lZpmwuK0SYVKRGlEXru2LxZV2ENk_Kb_f2sH5aLc_jzW7R3H-6_fHo_MMgx7J9ONW73ZPuwS6PDZlh9Qs; _cs_mk_aa=0.22721445995290246_1729076303401; nbatag_main_v_id=019294faa08d001b55502fa82f350506f001906700799; nbatag_main__sn=1; nbatag_main_ses_id=1729076306064%3Bexp-session; _li_dcdm_c=.nba.com; _lc2_fpi=b74e83586b48--01jaafn855twrf7zqbafy79z43; ug=670f9c53040c360a3f9daa0015e6e39a; ugs=1; ab.storage.deviceId.cf150dab-3153-49b0-b48c-66a7c18688ea=%7B%22g%22%3A%22d02e32c5-488d-04df-8a72-f02564e3074c%22%2C%22c%22%3A1729076307585%2C%22l%22%3A1729076307585%7D; s_cc=true; amCustPrevPage=null; _cs_c=0; umto=1; bea4r=670f9c6afafb2f0a3f9daa0015e6e39a; nbatag_main__ss=0%3Bexp-session; OptanonAlertBoxClosed=2024-10-16T10:58:59.817Z; eupubconsent-v2=CQGka8AQGka8AAcABBFRBLF8AP_gAAAAACiQKhQL4AFAAaABUADIAIAASAAqABaADIAGgAOoAiACKAEmAJgAnABbAC-AGEAQAAhABSADKAIAAQgAiwBHQCdgI1AUeAvMBiwDGQGzANqAbaA2-CcAJyQTmBOqCdgJ9AT8An_BQEFA4KDAomBRWCi4KMwUgBSuCloKaQU2BT-CoIKhAAAAJCQCAAFgAVABBADIANAAiABMAEIAvMIACAUeAxYdAIAAWABUAEEAMgA0ACIAEwAiwC8xwAIAhADFiEAEABZKAIAAsAIgATAF5kgAIDFikAgABYAFQAQQAyADQAIgATACLALzKAAQGLAA.f_wAAAAAAAAA; bounceClientVisit5454v=N4IgNgDiBcIBYBcEQM4FIDMBBNAmAYnvgO6kB0AdgEYCGZAxgPYC2RKCNC6B9ArswFN2nFCAA0IAE4wQpYpVoMW4kAEsUAfQDmjDSiEpVjCjABmNMPonrtEPQaMno5ywIC+QA; OptanonControl=ccc=CN&csc=&cic=1&otvers=202403.2.0&pctm=2024-10-16T10%3A58%3A59.817Z&reg=global&ustcs=1---&vers=4.1.4-nba; at_check=true; bea4=umw9118_7426362118066390605; goiz=2ee50266b45343308c8d672d1dea0f98; zwmc=6687193676502893530; orev=F; s_gpv_pageModal=nba%3Astats; aaCustPrevPage=nba:stats; nbaOrigin=Page%20View%3A%20Stats%7C%7CStats; aam_uuid=63355782796040799983390705050697018002; s_ips=674.8399963378906; _cs_id=f67ee6fd-165e-a66f-c842-f06e272165b0.1729076310.1.1729076789.1729076310.1.1763240310429.1; mbox=session#b6b06ace5f484816ba0563b9a24715e9#1729078651|PC#b6b06ace5f484816ba0563b9a24715e9.32_0#1792321591; s_tp=1990; s_ppv=nba%253Astats%2C114%2C34%2C2274%2C2%2C3; _cs_s=6.0.0.1729078887709; __gads=ID=ba2eaaf8f9f93dfe:T=1729076345:RT=1729077127:S=ALNI_MaM1-kKGxHKd2gskoZewtjbXmKJGQ; __gpi=UID=00000f4629d3a227:T=1729076345:RT=1729077127:S=ALNI_MZC_TXuIyyRliOQhUcIRaiLmcNFrA; __eoi=ID=efd493b5ca8a8fac:T=1729076345:RT=1729077127:S=AA-AfjaU6CKesnarbrd5_i-9mf3J; iframeRef=www.nba.com/stats/teams/traditional; bm_sz=C3D20791349C2322A775CB44E8FB74C6~YAAQy+nHF2qIaX+SAQAAYn4HlRloO4K0MOgyCMY25/OQq/jQRu2lKpna4q3BdhCmaXV3HCjgCj4bYA2D7bOidRfLQWtf7W4h8KZD2IxRkdWQe/R0YgA1nn+/3XuMM8sLimUNzDH0BlPEfvjZHoUjI21pXDTVLC2CIk3bkCfGEx1jcDrm6SBejPmVlk71GqUKkXBq0De4CKlV26nWktkUWJuV2mk3IbCI0BtTSn434nrQPVKlTRGzZ0qQussilL+L41Q4qdGB8g+b/1/f2LfIPK6LmIe3LaOPhqxH+xG2g2kBc+1Wc647GIX7zRzhNRqoFs4lLzfbViFnkNuLdkwYi+S62GqMDqXh59Bo1ASCOitsbx7k2BpAe1DUpASCdIVPGYZ5QvdU/nAXu2IWYxW1bDLExHZZQ2DEYMdSKvTgxlTzoosPbEYQ0iwCuj8Fag1/8BE0KOPVODSsXS2NykYFHG+vhEW+itOiwoX5JPXvgufmjxEp+5O713E4+HTuIOXrcDXVz5ps2yv0LuMmpIjgrVO6fASjk+dX~3491379~4274229; OptanonConsent=isGpcEnabled=0&datestamp=Wed+Oct+16+2024+19%3A12%3A30+GMT%2B0800+(%E4%B8%AD%E5%9B%BD%E6%A0%87%E5%87%86%E6%97%B6%E9%97%B4)&version=202403.2.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=beeec711-9e60-4b26-8f84-e4f2fb3470a3&interactionCount=2&isAnonUser=1&landingPath=NotLandingPage&groups=dsa%3A1%2Ccad%3A1%2CNBAad%3A1%2Cmcp%3A1%2CNBAmt%3A1%2Cpad%3A1%2Cpap%3A1%2Cgld%3A1%2Cpcd%3A1%2Cpcp%3A1%2Cmap%3A1%2Cmra%3A1%2Cpdd%3A1%2Csid%3A1%2Csec%3A1%2Ctdc%3A1%2Ccos%3A1%2Cdlk%3A1%2Cdid%3A1%2Cdsh%3A1%2Cdsl%3A1%2Cven%3A1%2Creq%3A1&AwaitingReconsent=false&intType=1&geolocation=CN%3B; ab.storage.sessionId.cf150dab-3153-49b0-b48c-66a7c18688ea=%7B%22g%22%3A%22d7bf48f3-d1cc-eac6-45a6-b287f752cac6%22%2C%22e%22%3A1729078950121%2C%22c%22%3A1729076307581%2C%22l%22%3A1729077150121%7D; nbatag_main__pn=18%3Bexp-session; amp_2442d5=GzgnMQi3z9xO0oIKIu6wzC.bnVsbA==..1iaafma2d.1iaagf1b2.g.11.1h; ak_bmsc=798A2D04AD30CC0EC6C72E8ADA255863~000000000000000000000000000000~YAAQ35o7F6UGbHqSAQAA0IsHlRkT1zREe9Qy4jownjTkXoLl2B01imkeLwZhzMjDrHQGbZCMe+otSmJWDYfpb66VE4N/UEiHo9UooUGJBIrKYWd6kHOlFM7ooKjienJ9OQUBxUkn9ghnFWY7wTxiWQOW7ySOwTR/fpa8wmndPMq4oq7qhs7sFTUjhigb8IjFLhQWwkNPynrzHU9oQq0gW2BWbuPcX13EUukdREmEoTDkAgr+rZ2FIf2gQ9N05gYdWqeWjMP4rB/C+ZaRQj4SmRbKthhb4PFEnF13thUrpE7x1Rg9+3RS+KTffTNe2uY+AXSNiAnAJkZdkBci3GN/Ph9GY4epa65+g7aHGVzOOv7xxYsg+NB2TVZcNjeNNlcFE/PQAx62pfpuP690pAl7cdFcFLlbhiyxnbhqlc7GSN11ctJrOFBxX0lQEg==; bm_sv=B5F18EAEDAB5A444DAAE96A0F4E8D9A7~YAAQ35o7F6YGbHqSAQAA0IsHlRlzDYXPWzr4BuLm2sTuQy0gJPnULpFAJovDvux06+k0jKo24ldN0/rPiaV4ddfBtwgGxgGBcmQxFzlqm/9XUx6IB6RecKawGIXzfESE0W2Gyy9ep/fXG5mXAtxpgJeJlIiVNAP6Dgs+5wsuGzCnKfVaU1dT7YoySnoXd7H53EskbE2/Ixt+9IyJmz9qvfJjKRuojWZw7WuBKCjxLeR3qiI+65Yx6SYB3EFiZg==~1; nbatag_main__se=92%3Bexp-session; nbatag_main__st=1729079241484%3Bexp-session; amp_2442d5_nba.com=GzgnMQi3z9xO0oIKIu6wzC.bnVsbA==..1iaafma2d.1iaagntk8.b.2l.30'
# }

# response = requests.get(url,headers=headers)
# content = response.text
# soup = BeautifulSoup(content, 'html.parser')

# table = soup.findAll('table', class_='Crom_table__p1iZz')
# print(table)
# from selenium import webdriver
# from bs4 import BeautifulSoup
# import time

# driver = webdriver.Chrome()  # 启动Chrome浏览器
# driver.get('https://www.nba.com/stats/teams/traditional?Season=1996-97&SeasonType=Playoffs&dir=A&sort=W')
# time.sleep(1)  # 等待页面完全加载完毕

# page_source = driver.page_source
# soup = BeautifulSoup(page_source, 'lxml')

# driver.quit()  # 关闭浏览器

# table = soup.findAll("Crom_table__p1iZz")
# print(table)

# from lxml import html
# tree = html.fromstring(html_content)

# # 使用XPath定位到表格
# table_xpath = '/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table'

# # 提取表头
# headers_xpath = './/thead/tr/th'
# headers = tree.xpath(table_xpath + headers_xpath)[0].xpath('./@field')

# # 提取表格数据
# rows_xpath = './/tbody/tr'
# rows = tree.xpath(table_xpath + rows_xpath)

# teams_data = []
# for row in rows:
#     cells = row.xpath('.//td')
#     if cells:
#         # 获取Team名称，根据实际列的位置来获取
#         team_name = cells[1].text_content().strip()
#         # 可以继续获取其他数据
#         data = {header: cell.text_content().strip() for header, cell in zip(headers, cells)}
#         teams_data.append(data)

# # 打印提取的数据
# for data in teams_data:
#     print(data)
# url = 'https://www.nba.com/stats/teams/traditional?Season=1996-97&SeasonType=Playoffs&dir=A&sort=W'
# res = requests.get(url)

# html = etree.HTML(res.text)
# rest = html.xpath('/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table/thead/tr')  # 返回Element对象
# print(rest)

# import requests
# from lxml import etree

# # 定义URL
# url = 'https://www.nba.com/stats/teams/traditional?Season=1996-97&SeasonType=Playoffs&dir=A&sort=W'

# # 发送HTTP请求
# res = requests.get(url)

# # 检查请求是否成功
# if res.status_code == 200:
#     # 使用lxml解析HTML
#     html = etree.HTML(res.text)
    
#     # 使用XPath定位到表格的表头
#     xpath_expression = '/html/body/div[1]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table/thead/tr'
#     thead = html.xpath(xpath_expression)
    
#     # 打印结果
#     print(thead)
# else:
#     print(f"Failed to retrieve the webpage. Status code: {res.status_code}")


