# ICD9 GROUPS
cohort['icd9_group'] = 0

icd_list = set(cohort['icd9_code'])
icd_group = set()
for icd in icd_list:
    if "V" in icd:
        index = cohort.loc[cohort['icd9_code'] == icd].index
        cohort.at[index, 'icd9_group'] = 18
        icd_group.add(18)
    elif "E" in icd:
        index = cohort.loc[cohort['icd9_code'] == icd].index
        cohort.at[index, 'icd9_group'] = 19
        icd_group.add(19)
    else:
        if len(icd) == 4:
            icd9 = int(icd[:-1])
        elif len(icd) == 5:
            icd9 = int(icd[:-2])

        if icd9 <= 139:
            # Infectious And Parasitic Diseases
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 1
            icd_group.add(1)
        elif icd9 <= 239:
            # Neoplasms
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 2
            icd_group.add(2)
        elif icd9 <= 279:
            # Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 3
            icd_group.add(3)
        elif icd9 <= 289:
            # Diseases Of The Blood And Blood-Forming Organs
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 4
            icd_group.add(4)
        elif icd9 <= 319:
            # Mental Disorders
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 5
            icd_group.add(5)
        elif icd9 <= 389:
            # Diseases Of The Nervous System And Sense Organs
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 6
            icd_group.add(6)
        elif icd9 <= 459:
            #  Diseases Of The Circulatory System
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 7
            icd_group.add(7)
        elif icd9 <= 519:
            #  Diseases Of The Respiratory System
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 8
            icd_group.add(8)
        elif icd9 <= 579:
            #  Diseases Of The Digestive System
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 9
            icd_group.add(9)
        elif icd9 <= 629:
            #  Diseases Of The Genitourinary System
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 10
            icd_group.add(10)
        elif icd9 <= 679:
            #  Complications Of Pregnancy, Childbirth, And The Puerperium
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 11
            icd_group.add(11)
        elif icd9 <= 709:
            #  Diseases Of The Skin And Subcutaneous Tissue
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 12
            icd_group.add(12)
        elif icd9 <= 739:
            #  Diseases Of The Musculoskeletal System And Connective Tissue
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 13
            icd_group.add(13)
        elif icd9 <= 759:
            #  Congenital Anomalies
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 14
            icd_group.add(14)
        elif icd9 <= 779:
            #  Certain Conditions Originating In The Perinatal Period
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 15
            icd_group.add(15)
        elif icd9 <= 799:
            #  Symptoms, Signs, And Ill-Defined Conditions
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 16
            icd_group.add(16)
        elif icd9 <= 999:
            #  Injury And Poisoning
            index = cohort.loc[cohort['icd9_code'] == icd].index
            cohort.at[index, 'icd9_group'] = 17
            icd_group.add(17)