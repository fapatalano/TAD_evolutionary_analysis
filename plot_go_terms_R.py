#python import

import os

import matplotlib.pyplot as plt
import pandas as pd
import re
import textwrap
import seaborn as sns
import math
import numpy as np

# R import
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri

def get_cynteny_df(aln_name, aln_path='/Users/fabianpa/Desktop/new_sequences/synteny_3/'):
    aln_fname = os.path.join(aln_path, aln_name + ".synteny")
    with open(aln_fname, 'r') as synteny_file:
        synteny = synteny_file.read()
        species = [s.split(".coord")[0].split('/')[-1] for s in synteny.split('\n', 1)[0].split()[1:]]
        found_aln = re.findall(r"Alignment*", synteny, re.MULTILINE)
        n_aln = list(map(lambda x, y: x + str(y), found_aln, range(1, len(found_aln) + 1)))
        all_aln = {key: {} for key in n_aln}
        len_row = len(synteny.split('Ali')[1].split('\n')[1].split())
        n_keys = list(range(1, len_row, 2))
        for i in synteny.split('Ali'):
            aln = None
            for j in i.split('\n'):
                if j.startswith('gnment '):
                    aln = 'Alignment' + j.split()[1]
                    all_aln[aln] = {key: [] for key in n_keys}
                elif aln:
                    [all_aln[aln][idx].append(j.strip().split()[idx].split('_')[0]) for idx in
                     range(len(j.strip().split())) if idx % 2 != 0]
    return all_aln, species

#     converted_gene = r.bitr(gene, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
#     r.head(converted_gene.rx2("ENTREZID"))
#     # orgDb=get_db()
    # ego = clusterProfiler.enrichGO( gene          = gene,
    #                                 keyType       = "SYMBOL",
    #                                 OrgDb         = "org.Hs.eg.db",
    #                                 ont           = "BP",
    #                                 pAdjustMethod = "BH",
    #                                 pvalueCutoff  = 0.05,
    #                                 qvalueCutoff  = 0.05,
    #                                 readable      = True)
    # upSimGO =  clusterProfiler.simplify(ego, cutoff=0.7, by="p.adjust", select_fun=r.min, measure="Wang",
    #                    semData=ro.NULL)
    # return  upSimGO
# ggo <- groupGO(gene     = gene_list,
#                keyType       = "SYMBOL",
#                OrgDb    = db,
#                ont      = "BP",
#                level    = 4,
#                readable = TRUE)
# upSimGO = simplify(ggo, cutoff=0.6, by="Count", select_fun=min, measure="Wang")
# print(upSimGO)
# df = as.data.frame(ggo)
# df_sorted = df[order(-df$Count), ]
# png(output, width = 800, height = 800)
#
# # p <- barplot(as.numeric(df_sorted[1:10, ]$Count),names.arg = df_sorted[1:10, ]$Description,horiz = T,las=1,col="steelblue",xlab="Count")
# p <- barplot(upSimGO, showCategory=20,drop=TRUE)
# print(p)
# dev.off()

r_code = """ 
        library(org.Hs.eg.db)   #human
        library(org.Mmu.eg.db)  #rhesus
        library(org.Mm.eg.db)   #mouse
        library(org.Rn.eg.db)   #rat
                                #rabbit
        library(org.Bt.eg.db)   #cow
                                #sheep
        library(org.Ss.eg.db)   #pig
                                #cat
        library("org.Cf.eg.db") #dog
        library("org.Gg.eg.db") #chicken
        library("org.Dr.eg.db") #zebrafish
        
        library(clusterProfiler)
        library(enrichplot)
        library(ggplot2)
        library(cowplot)
        
        analyze_gene_list_count <- function(gene_list,db){
            ggo <- groupGO(gene     = gene_list,
                               keyType  = "SYMBOL",
                               OrgDb    = db,
                               ont      = "BP",
                               level    = 2,
                               readable = TRUE)
            return(ggo) 
        }
        
        analyze_gene_list_enr <- function(gene_list,db,pval){
            ego = enrichGO( gene          = gene_list,
                            keyType       = "SYMBOL",
                            OrgDb         =  db,
                            ont           = "BP",
                            pAdjustMethod = "BH",
                            pvalueCutoff  = pval, 
                            qvalueCutoff  = pval,
                            readable      = TRUE)
            upSimGO = simplify(ego, cutoff=0.7, by="p.adjust", select_fun=min, measure="Wang", semData=NULL)
            return(upSimGO) 
        }
            
        plot_result <- function(ego,filename){
            plot <- dotplot(ego,showCategory=10,font.size=10,label_format=70)
            ggsave(filename, plot,width = 10, height = 6)
        }
        
        """
def plot_bar(df):
    # Function to wrap text
    def wrap_labels(labels, width=30):
        return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

    wrapped_labels = wrap_labels(df['Description'])
    plt.figure(figsize=(12, 6))
    plt.barh(wrapped_labels, df['Count'])
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('')
    plt.title('Counts of Gene Ontology', fontsize=14)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top
    plt.tight_layout()  # Adjust layout to make room for the labels
    plt.show()

def plot_dotplot(dfs,output):

    def wrap_labels(labels, width=20):
        return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

    species = dfs['species'].unique()
    dfs['GeneRatio'] = dfs['GeneRatio'].apply(lambda x: int(x.split('/')[0]) / int(x.split('/')[1]))
    dfs['Description'] = wrap_labels(dfs['Description'])

    rows = math.floor(math.sqrt(len(species)))
    columns = math.ceil(len(species) / rows)
    nplots = rows*columns
    N = nplots-len(species)
    fig,axs = plt.subplots(nrows=rows, ncols=columns,figsize=(35,22))
    norm = plt.Normalize(dfs['p.adjust'].min(), dfs['p.adjust'].max())
    sm = plt.cm.ScalarMappable(cmap="bwr", norm=norm)

    all_handles = []
    all_labels = []
    label_index = []
    for i,s in enumerate(species):
        ax = axs.flatten()[i]
        df = dfs.loc[dfs["species"]==s]
        scatter = sns.scatterplot(data=df,y="Description", x="GeneRatio", hue='p.adjust',size="Count",palette="bwr",hue_norm= sm.norm,sizes=(200, 500),ax=ax)
        ax.set_xlabel('GeneRatio', fontsize=20)
        ax.set_ylabel('')
        ax.tick_params(labelsize=20)
        handles, labels = scatter.get_legend_handles_labels()
        all_handles.extend(handles)
        label_index.extend(labels)
        all_labels.extend(df['Count'].astype(str).values)
        ax.get_legend().remove()

    all_labels = ["Count"]+sorted(list(set(all_labels)))

    indices= [list(range(i,i+3)) for i in np.where(np.array(label_index) == "Count")[0] if label_index[i:i+3] == all_labels][0]
    handle= [all_handles[i] for i  in indices]
    cbar = fig.colorbar(sm, ax=axs,shrink= .3,aspect=5,anchor=(-3,-.1))
    cbar.set_label('p.adjust')
    fig.legend(handle, all_labels, bbox_to_anchor=(.8,.3), frameon=False, markerscale=3,labelspacing=3, prop={'size': 20})
    if N!=0: [ax.remove() for ax in axs.flat[-N:]]
    plt.tight_layout()
    plt.savefig("../images/figure 6/go_bm.svg")
    plt.show()

def get_go_in_sb(df,org):
    alignment = ['Alignment12', 'Alignment130', 'Alignment134', 'Alignment141', 'Alignment149', 'Alignment197', 'Alignment200', 'Alignment218',
                 'Alignment220', 'Alignment247', 'Alignment254', 'Alignment256', 'Alignment275', 'Alignment293', 'Alignment295', 'Alignment304',
                 'Alignment305', 'Alignment315', 'Alignment321', 'Alignment337', 'Alignment351', 'Alignment353', 'Alignment375', 'Alignment377',
                 'Alignment392', 'Alignment401', 'Alignment419', 'Alignment426', 'Alignment431', 'Alignment438', 'Alignment464', 'Alignment493',
                 'Alignment498', 'Alignment68', 'Alignment78', 'Alignment95']

    # alignment = ['Alignment1','Alignment10','Alignment100','Alignment101','Alignment102','Alignment103','Alignment104','Alignment105','Alignment106','Alignment107',
    #              'Alignment108','Alignment109','Alignment11','Alignment110','Alignment111','Alignment112','Alignment113','Alignment114','Alignment115','Alignment116',
    #              'Alignment117','Alignment118','Alignment119','Alignment120','Alignment121','Alignment122','Alignment123','Alignment124','Alignment125','Alignment126','Alignment127','Alignment128','Alignment129','Alignment13','Alignment131','Alignment132','Alignment133','Alignment135','Alignment136','Alignment137','Alignment138','Alignment139','Alignment14','Alignment140','Alignment142','Alignment143','Alignment144','Alignment145','Alignment146','Alignment147','Alignment148','Alignment15','Alignment150','Alignment151','Alignment152','Alignment153','Alignment154','Alignment155','Alignment156','Alignment157','Alignment158','Alignment159','Alignment16','Alignment160','Alignment162','Alignment163','Alignment164','Alignment165','Alignment166','Alignment167','Alignment168','Alignment169','Alignment17','Alignment170','Alignment171','Alignment172','Alignment174','Alignment175','Alignment176','Alignment177','Alignment178','Alignment179','Alignment18','Alignment180','Alignment181','Alignment182','Alignment183','Alignment184','Alignment185','Alignment186','Alignment187','Alignment188','Alignment189','Alignment19','Alignment190','Alignment191','Alignment192','Alignment193','Alignment194','Alignment195','Alignment196','Alignment198','Alignment199','Alignment2','Alignment20','Alignment201','Alignment202','Alignment203','Alignment204','Alignment205','Alignment206','Alignment207','Alignment208','Alignment209','Alignment21','Alignment210','Alignment211','Alignment212','Alignment213','Alignment214','Alignment215','Alignment216','Alignment217','Alignment219','Alignment22','Alignment221','Alignment222','Alignment223','Alignment224','Alignment225','Alignment226','Alignment227','Alignment228','Alignment229','Alignment23','Alignment230','Alignment231','Alignment232','Alignment233','Alignment234','Alignment235','Alignment236','Alignment237','Alignment238','Alignment239','Alignment24','Alignment240','Alignment241','Alignment242','Alignment243','Alignment244','Alignment245','Alignment246','Alignment248','Alignment25','Alignment250','Alignment251','Alignment252','Alignment253','Alignment255','Alignment257','Alignment258','Alignment26','Alignment260','Alignment261','Alignment262','Alignment263','Alignment264','Alignment265','Alignment266','Alignment267','Alignment268','Alignment269','Alignment27','Alignment270','Alignment271','Alignment272','Alignment273','Alignment274','Alignment276','Alignment277','Alignment278','Alignment279','Alignment28','Alignment280','Alignment281','Alignment282','Alignment283','Alignment284','Alignment285','Alignment286','Alignment287','Alignment288','Alignment289','Alignment29','Alignment291','Alignment292','Alignment296','Alignment297','Alignment298','Alignment299','Alignment3','Alignment30','Alignment300','Alignment301','Alignment303','Alignment306','Alignment307','Alignment308','Alignment309','Alignment31','Alignment310','Alignment311','Alignment312','Alignment313','Alignment314','Alignment316','Alignment317','Alignment318','Alignment319','Alignment32','Alignment320','Alignment322','Alignment323','Alignment324','Alignment325','Alignment326','Alignment327','Alignment328','Alignment329','Alignment33','Alignment330','Alignment331','Alignment332','Alignment333','Alignment334','Alignment336','Alignment338','Alignment339','Alignment34','Alignment340','Alignment341','Alignment342','Alignment343','Alignment344','Alignment345','Alignment346','Alignment347','Alignment348','Alignment349','Alignment35','Alignment350','Alignment352','Alignment354','Alignment356','Alignment357','Alignment358','Alignment359','Alignment36','Alignment360','Alignment361','Alignment362','Alignment363','Alignment364','Alignment365','Alignment367','Alignment368','Alignment369','Alignment37','Alignment370','Alignment371','Alignment372','Alignment373','Alignment374','Alignment376','Alignment378','Alignment379','Alignment38','Alignment380','Alignment381','Alignment382','Alignment383','Alignment384','Alignment385','Alignment386','Alignment387','Alignment389','Alignment39','Alignment390','Alignment391','Alignment395','Alignment396','Alignment397','Alignment399','Alignment4','Alignment40','Alignment400','Alignment402','Alignment403','Alignment404','Alignment405','Alignment406','Alignment407','Alignment408','Alignment41','Alignment410','Alignment411','Alignment413','Alignment414','Alignment415','Alignment416','Alignment417','Alignment418','Alignment42','Alignment420','Alignment421','Alignment422','Alignment425','Alignment427','Alignment428','Alignment429','Alignment43','Alignment430','Alignment432','Alignment433','Alignment434','Alignment435','Alignment436','Alignment437','Alignment439','Alignment44','Alignment440','Alignment441','Alignment442','Alignment443','Alignment444','Alignment445','Alignment447','Alignment449','Alignment45','Alignment450','Alignment451','Alignment455','Alignment456','Alignment457','Alignment458','Alignment459','Alignment46','Alignment460','Alignment462','Alignment463','Alignment465','Alignment466','Alignment467','Alignment468','Alignment47','Alignment472','Alignment473','Alignment474','Alignment475','Alignment476','Alignment477','Alignment48','Alignment480','Alignment481','Alignment482','Alignment483','Alignment484','Alignment485','Alignment486','Alignment487','Alignment489','Alignment49','Alignment490','Alignment491','Alignment492','Alignment494','Alignment495','Alignment496','Alignment497','Alignment499','Alignment5','Alignment50','Alignment500','Alignment502','Alignment503','Alignment504','Alignment507','Alignment508','Alignment51','Alignment510','Alignment511','Alignment513','Alignment515','Alignment518','Alignment519','Alignment52','Alignment522','Alignment523','Alignment524','Alignment526','Alignment528','Alignment53','Alignment54','Alignment55','Alignment56','Alignment57','Alignment58','Alignment59','Alignment6','Alignment60','Alignment61','Alignment62','Alignment63','Alignment64','Alignment65','Alignment66','Alignment67','Alignment69','Alignment7','Alignment70','Alignment71','Alignment72','Alignment73','Alignment74','Alignment75','Alignment76','Alignment77','Alignment79','Alignment8','Alignment80','Alignment81','Alignment82','Alignment84','Alignment85','Alignment86','Alignment87','Alignment88','Alignment89','Alignment9','Alignment90','Alignment91','Alignment92','Alignment93','Alignment94','Alignment96','Alignment97','Alignment98','Alignment99']
    # genes = df.xs(org, level=1, axis=1).values.flatten()
    genes = df.xs(org, level=1, axis=1)[alignment].values.flatten()
    filtered_genes = [x for x in genes if not pd.isna(x) and x!='-']
    return filtered_genes

def get_db(species):
    sp_to_db = {"human":"org.Hs.eg.db",
                "rhesus":"org.Hs.eg.db",
                # "rhesus":"org.Mmu.eg.db",
                "mouse":"org.Hs.eg.db",
                # "mouse":"org.Mm.eg.db",
                "rat":"org.Hs.eg.db",
                # "rat":"org.Rn.eg.db",
                "rabbit": "org.Hs.eg.db",
                "cow": "org.Hs.eg.db",
                # "cow":"org.Bt.eg.db",
                "sheep": "org.Hs.eg.db",
                # "pig":"org.Ss.eg.db",
                "pig":"org.Hs.eg.db",
                "cat":"org.Hs.eg.db",
                "dog":"org.Hs.eg.db",
                # "dog":"org.Cf.eg.db",
                "chicken":"org.Hs.eg.db",
                # "chicken":"org.Gg.eg.db",
                "zfish":"org.Hs.eg.db"
                # "zfish":"org.Dr.eg.db"
                }
    return sp_to_db.get(species)

def main(species_aln):
    r(r_code)
    pandas2ri.activate()
    all_aln, species = get_cynteny_df(species_aln)
    reform = {(outerKey, innerKey): values for outerKey, innerDict in all_aln.items() for innerKey, values in
              innerDict.items()}
    df = pd.DataFrame.from_dict(reform, orient='index').transpose()
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    first = list(all_aln.keys())[0]
    column_n = df[first].columns.values
    columns_name = {column_n[i]: species[i] for i in range(len(column_n))}
    df.rename(columns=columns_name, inplace=True)
    result = []
    for org in species:
        print(org)
        # plot_type = "enr_ou_proc"
        plot_type = "enr_bm_proc"
        # plot_type= "bar_bm_proc"
        gene_list = get_go_in_sb(df,org)
        gene_vector = ro.StrVector(gene_list)
        output=f'../images/figure 6/go_{org}_{aln}_{plot_type}.png'
        db =get_db(org)

        if plot_type.__contains__("bar"):
            ego = ro.r['analyze_gene_list_count'](gene_vector, db)
            ego_df = ro.conversion.rpy2py(ego.slots['result'])
            filtered_ego_df = ego_df.loc[ego_df["Count"]!=0].sort_values(by=["Count"],ascending=False)
            plot_bar(filtered_ego_df[10:20])
        else:
            # if org in ["cat", 'mouse', 'rabbit']:  ego = ro.r['analyze_gene_list_enr'](gene_vector, db,.15)
            # else:  ego = ro.r['analyze_gene_list_enr'](gene_vector, db,.05)
            if org =="human":
                ego = ro.r['analyze_gene_list_enr'](gene_vector, db, .05)
                ro.r['plot_result'](ego, output)

            # ego_df = ro.conversion.rpy2py(ego.slots['result'])
            # ego_df['species'] = org
            # result.append(ego_df)

    # resuld_df = pd.concat(result)
    # plot_dotplot(resuld_df,output)
    # ro.r['plot_result'](ego,output)

if __name__ == "__main__":
    aln = "hrmrrcspcd"
    main(aln)