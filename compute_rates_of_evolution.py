import os
import re

from tqdm import tqdm

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import bioframe as bf
from scipy.stats import linregress

from io import StringIO
from Bio import Phylo

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
sns.set_context("paper")

def get_code_conversion(code):
    alignment_code_to_species = {'hr':          'human;rhesus',
                                 'mr':          'mouse;rat',
                                 'cd':          'cat;dog',
                                 'cs':          'cow;sheep',
                                 'csp':         'pig;cow;sheep',
                                 'mrr':         'mouse;rabbit;rat',
                                 'cspcd':       'pig;cow;sheep;dog;cat',
                                 'hrmrr':       'human;mouse;rhesus;rabbit;rat',
                                 'hrmrrcspcd':  'human;mouse;rhesus;rabbit;rat;pig;cow;sheep;dog;cat',
                                 'hrmrrcspcdc': 'human;mouse;rhesus;rabbit;rat;pig;cow;sheep;dog;cat;chicken',
                                 'hrmrrcspcdcz':'human;mouse;rhesus;rabbit;rat;pig;cow;sheep;dog;cat;chicken;zfish'}
    return alignment_code_to_species.get(code)

def get_sb_len(species, aln, path="/Users/fabianpa/Desktop/new_sequences/synteny_3/synteny_bed"):
    sb_dfs = []
    for common_name in species:
        fname = os.path.join(path, common_name + "_" + aln + ".bed")
        sb_df = pd.read_table(fname, names=("chr", "start", "end", "aln", "strand"), header=None)
        sb_df["species"] = common_name
        sb_df["sb_length"] = sb_df["end"] - sb_df["start"]
        sb_dfs.append(sb_df)
    return pd.concat(sb_dfs, axis=0)

def get_tad_in_sb(species, aln, sb_coord, sb_specific_aln=None, shuffle=False,
                  coord_path="/Users/fabianpa/Desktop/new_sequences/sb_in_tad_3/"):
    count_df = []
    coord_df = []
    for common_name in species:
        coord_fname = os.path.join(coord_path, f"{common_name.lower()}_{aln}.tad.sb")
        specie_aln_name = common_name.lower() + "_" + aln
        try:
            sb_tad_coord = \
            pd.read_csv(coord_fname, sep="\t", usecols=[5, 6, 7, 3, 4], names=("aln", "strand", "chrom", "start", "end"),
                        header=None)[["chrom", "start", "end", "aln", "strand"]]
            sb_tad_coord["species"] = common_name
            sb_tad_coord["tad_size"] = sb_tad_coord["end"] - sb_tad_coord["start"]
            if sb_specific_aln:
                coord_df.append(sb_tad_coord[sb_tad_coord.aln.isin(sb_specific_aln[specie_aln_name])])
            else:
                coord_df.append(sb_tad_coord)
            count_list = []
            for i in sb_coord.aln.unique():
                count = len(sb_tad_coord[(sb_tad_coord.aln == i)])
                count_list.append([i, count, common_name])

            if sb_specific_aln:
                df = pd.DataFrame(count_list, columns=['aln', 'count', "species"])
                if shuffle: df["count"] = df["count"].sample(frac=1).values
                count_df.append(df[df["aln"].isin(sb_specific_aln[specie_aln_name])])
            else:
                count_df.append(pd.DataFrame(count_list, columns=['aln', 'count', "species"]))

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e.filename}")
    return count_df, coord_df

def get_cynteny_df(aln_name, aln_path='/Users/fabianpa/Desktop/new_sequences/synteny_3/'):
    aln_fname = os.path.join(aln_path, aln_name + ".synteny")
    with open(aln_fname, 'r') as synteny_file:
        synteny = synteny_file.read()
        species = [s.split(".coord")[0].split('/')[-1] for s in synteny.split('\n', 1)[0].split()[1:]]
        found_aln = re.findall(r"Alignment*", synteny, re.MULTILINE)
        n_aln = list(map(lambda x, y: x + str(y), found_aln, range(1, len(found_aln) + 1)))
        all_aln = {key: {} for key in n_aln}

        n_keys = list(range(len(synteny.split('Ali')[1].split('\n')[1].split())))
        for i in synteny.split('Ali'):
            aln = None
            for j in i.split('\n'):
                if j.startswith('gnment '):
                    aln = 'Alignment' + j.split()[1]
                    all_aln[aln] = {key: [] for key in n_keys}
                elif aln:
                    [all_aln[aln][idx].append(j.strip().split()[idx]) for idx in range(len(j.strip().split()))]
    return all_aln, species

def read_coord_file(fname):
    with open(fname, 'r') as coord:
        coord_file = coord.readlines()[1:]
        coord_dict = {
            'gene_name': [i.strip().split()[0] for i in coord_file if len(i.strip().split()) == 5],
            'chrom': [i.strip().split()[1] for i in coord_file if len(i.strip().split()) == 5],
            'start': [int(i.strip().split()[2]) for i in coord_file if len(i.strip().split()) == 5],
            'end': [int(i.strip().split()[3]) for i in coord_file if len(i.strip().split()) == 5],
            'strand': [i.strip().split()[4] for i in coord_file if len(i.strip().split()) == 5]}
    coord.close()
    return coord_dict

def get_equal_tad_count_syn_blocks(tad_count, species):
    s1_count = tad_count.loc[(tad_count.species == species[0])]
    s2_count = tad_count.loc[(tad_count.species == species[1])]

    count_diff = s1_count["count"] - s2_count["count"]
    count_diff_df = pd.DataFrame({"aln": s1_count.aln, "count_s1": s1_count["count"],
                                  "count_s2": s2_count["count"], "count_diff": count_diff})

    return count_diff_df.loc[
        (count_diff_df["count_diff"] == 0) & (count_diff_df["count_s1"] != 0), "aln"].values.tolist()

def prepare_gene_coords(gene_coords, species_name, gene_inters):
        coords = pd.DataFrame(gene_coords[species_name])
        coords['gene_name'] = coords['gene_name'].str.split('_').str[0]
        coords = coords[coords['gene_name'].isin(gene_inters)]

        coords.loc[coords['strand'] == '+', 'end'] = coords['start'] + 1
        coords.loc[coords['strand'] == '-', 'start'] = coords['end']
        coords.loc[coords['strand'] == '-', 'end'] = coords['end'] + 1

        coords['chrom'] = 'chr' + coords['chrom'].astype(str)
        return coords

def prepare_tads_for_species(tads, species_name, aln_number):
    strand = tads.loc[(tads.species == species_name) & (tads.aln == aln_number), "strand"].values[0]
    tad_df = tads.loc[(tads.species == species_name) & (tads.aln == aln_number)].sort_values(
        by=['start'], ascending=(strand != "-")).reset_index(drop=True)
    tad_df['tad_number'] = tad_df.index + 1
    return tad_df

def map_tad_boundaries(comb_df, syn_block, species):
    filtered_tad_coord = comb_df[
        (comb_df["aln"].isin(syn_block)) &
        (comb_df["species"].isin(species))
        ].copy()

    chr_y = filtered_tad_coord.loc[filtered_tad_coord.chrom == "chrY", "aln"].unique()
    filtered_tad_coord = filtered_tad_coord[~filtered_tad_coord['aln'].isin(chr_y)]

    tad_s1 = filtered_tad_coord.loc[(filtered_tad_coord.species == species[0]), ["start", "end", "aln"]]
    tad_s2 = filtered_tad_coord.loc[(filtered_tad_coord.species == species[1]), ["start", "end", "aln"]]

    scaler = MinMaxScaler(feature_range=(0, 1))
    sb = tad_s1['aln'].unique()

    s1_map = [
        np.unique(
            scaler.fit_transform(
                tad_s1.loc[tad_s1['aln'] == block, ['start', 'end']]
                .sort_values(by='start')
                .values.flatten().reshape(-1, 1)
            )
        )[1:-1].tolist()
        for block in sb
        if len(tad_s1.loc[tad_s1['aln'] == block, ['start', 'end']].sort_values(by='start')) > 1
    ]

    s2_map = [
        np.unique(
            scaler.fit_transform(
                (tad_s2.loc[tad_s2['aln'] == block, ['start', 'end']]
                .sort_values(by='start')
                .values.flatten().reshape(-1, 1))
            )
        )[1:-1].tolist()
        for block in sb
        if len(tad_s2.loc[tad_s2['aln'] == block, ['start', 'end']].sort_values(by='start')) > 1
    ]

    x = [item for sublist in s1_map for item in sublist]
    y = [item for sublist in s2_map for item in sublist]

    res = stats.linregress(x, y)
    r2 = res.rvalue ** 2
    p_val = res.pvalue
    return r2, p_val

def compute_delta_borders(comb_df, syn_block, species):

    # Filter TAD coordinates based on synteny block and species
    filtered_tad_coord = comb_df[
        (comb_df["aln"].isin(syn_block)) &
        (comb_df["species"].isin(species))
        ].copy()

    chr_y = filtered_tad_coord.loc[filtered_tad_coord.chrom == "chrY", "aln"].unique()
    filtered_tad_coord = filtered_tad_coord[~filtered_tad_coord['aln'].isin(chr_y)]

    tad_s1 = filtered_tad_coord.loc[(filtered_tad_coord.species == species[0]), ["start", "end", "aln"]]
    tad_s2 = filtered_tad_coord.loc[(filtered_tad_coord.species == species[1]), ["start", "end", "aln"]]

    # Sort and drop duplicates
    tad_s1 = tad_s1.sort_values(by=['aln', 'start']).drop_duplicates(['aln', 'start', 'end'])
    tad_s2 = tad_s2.sort_values(by=['aln', 'start']).drop_duplicates(['aln', 'start', 'end'])

    scaler = MinMaxScaler(feature_range=(0, 1))
    deltas = []
    block = tad_s1['aln'].unique()[0]
    block_tad_s1 = np.unique(tad_s1.loc[tad_s1['aln'] == block, ['start', 'end']].sort_values(by='start').values.flatten()).reshape(-1, 1)
    block_tad_s2 = np.unique(tad_s2.loc[tad_s2['aln'] == block, ['start', 'end']].sort_values(by='start').values.flatten()).reshape(-1, 1)

    if len(block_tad_s1) > 1:

        # Scale boundaries within each block
        scaled_tad_s1 = scaler.fit_transform(block_tad_s1)
        scaled_tad_s2 = scaler.fit_transform(block_tad_s2)
        for s1, s2 in zip(scaled_tad_s1, scaled_tad_s2):
            if s1!=0 and s1!=1:
                delta = abs(s1[0] - s2[0])
                deltas.append(delta)

    return np.median(deltas)

def calculate_gene_per_tad(s1,s2,genes_in_aln,gene_coords,tads,aln_number,max_length):
    gene_inters = set(genes_in_aln[s1].str.split('_').str[0]).intersection(
        set(genes_in_aln[s2].str.split('_').str[0]))
    gene_inters.discard('-')
    if gene_inters:
        gene_coords_s1 = prepare_gene_coords(gene_coords, s1, gene_inters)
        gene_coords_s2 = prepare_gene_coords(gene_coords, s2, gene_inters)

        tad_s1 = prepare_tads_for_species(tads, s1, aln_number)
        tad_s2 = prepare_tads_for_species(tads, s2, aln_number)

        gene_tad_overlap_s1 = bf.overlap(tad_s1, gene_coords_s1, return_overlap=True).dropna()
        gene_tad_overlap_s2 = bf.overlap(tad_s2, gene_coords_s2, return_overlap=True).dropna()

        # Prepare overlap data for merging by ensuring consistent column types
        gene_tad_overlap_s1['gene_name_'] = gene_tad_overlap_s1['gene_name_'].astype(str)
        gene_tad_overlap_s1['tad_number'] = gene_tad_overlap_s1['tad_number'].astype(str)

        gene_tad_overlap_s2['gene_name_'] = gene_tad_overlap_s2['gene_name_'].astype(str)
        gene_tad_overlap_s2['tad_number'] = gene_tad_overlap_s2['tad_number'].astype(str)

        merged_df = pd.merge(gene_tad_overlap_s1, gene_tad_overlap_s2,
                             on=['gene_name_', 'tad_number'], how='inner')

        max_genes = max(len(genes_in_aln[s1]),len(genes_in_aln[s2]))
        return len(merged_df)/max_length, max_genes

    return 0,0

def calculate_edit_per_block(s1,s2,genes_in_aln,gene_coords,tads,aln_number,max_length):

    gene_inters = set(genes_in_aln[s1].str.split('_').str[0]).intersection(
        set(genes_in_aln[s2].str.split('_').str[0]))
    gene_inters.discard('-')
    if gene_inters:
        gene_coords_s1 = prepare_gene_coords(gene_coords, s1, gene_inters)
        gene_coords_s2 = prepare_gene_coords(gene_coords, s2, gene_inters)

        tad_s1 = prepare_tads_for_species(tads, s1, aln_number)
        tad_s2 = prepare_tads_for_species(tads, s2, aln_number)

        gene_tad_overlap_s1 = bf.overlap(tad_s1, gene_coords_s1, return_overlap=True).dropna()
        gene_tad_overlap_s2 = bf.overlap(tad_s2, gene_coords_s2, return_overlap=True).dropna()

        # Group genes by TAD and convert to dictionary
        genes_by_tad_s1 = gene_tad_overlap_s1.groupby('tad_number')['gene_name_'].apply(list).to_dict()
        genes_by_tad_s2 = gene_tad_overlap_s2.groupby('tad_number')['gene_name_'].apply(list).to_dict()

        # Calculate insertions and deletions based on gene distributions across TADs
        insertions, deletions = 0, 0
        gene_distributions = {0:genes_by_tad_s1,1:genes_by_tad_s2}
        value_to_key_mapping = {value: key for key, values in gene_distributions[1].items() for value in values}
        result = {key: set([value_to_key_mapping[value] for value in values if value in value_to_key_mapping]) for key, values in gene_distributions[0].items()}
        insertions = sum(len(v) - 1 for k, v in result.items() if v)

        deletions = sum(
            len(value1 & value2) for key1, value1 in result.items() for key2, value2 in result.items() if
            key1 < key2 and len(value1 & value2) > 0)

        total_edits = deletions + insertions
        n_borders = max(len(tad_s1),len(tad_s2))
        return total_edits/max_length,n_borders+1
    else: return None,None

def get_divergence_time(species1, species2, tree):
    clade1 = tree.find_any(name=species1)
    clade2 = tree.find_any(name=species2)
    # mrca = tree.common_ancestor(clade1, clade2)
    # divergence_time = tree.distance(clade1, mrca) + tree.distance(clade2, mrca)
    divergence_time = tree.distance(species1, species2)/2
    return divergence_time

def load_species_data(newick_str,aln_code, coord_path):
    tree = Phylo.read(StringIO(newick_str), "newick")
    species_list = get_code_conversion(aln_code).split(';')
    sb_coord = get_sb_len(species_list, aln_code)
    tad_sb_list, tad_sb_coord_list = get_tad_in_sb(species_list, aln_code, sb_coord)
    tad_sb_df = pd.concat(tad_sb_list, axis=0).sort_values(by='aln')
    tad_sb_coord_df = pd.concat(tad_sb_coord_list, axis=0)
    species_coords = {f: read_coord_file(os.path.join(coord_path, f + ".coord")) for f in species_list}
    return tree,species_list, tad_sb_df, tad_sb_coord_df, species_coords

def finalize_rates(rate_dict, div_time_dict, col_name,type):
    if type == "conformation":
        rate_dict = {key: np.mean(value) for key, value in rate_dict.items() if value}
    else:
        rate_dict = {key: np.nanmedian(value) for key, value in rate_dict.items() if value}

    rate_df = pd.DataFrame.from_dict(rate_dict, orient='index', columns=[col_name])
    rate_df['divergence_time'] = rate_df.index.map(div_time_dict)
    return rate_df

def calculate_rates(type,tad_sb_df,tad_sb_coord_df,species,tree,sb_coord,species_coords,df):

    comb = list(combinations(species, 2))
    div_time_dict = {}
    rate_per_mb = {i: [] for i in comb}
    density = {i: [] for i in comb}

    for aln in tqdm(tad_sb_df["aln"].unique()[:], desc="Block analysed", colour="green"):
        for species1, species2 in comb[:]:
            count_rate_diff = None
            sb_len_s1 = sb_coord.loc[(sb_coord['species'] == species1) & (sb_coord['aln'] == aln), "sb_length"].values[
                0] / 1e6
            sb_len_s2 = sb_coord.loc[(sb_coord['species'] == species2) & (sb_coord['aln'] == aln), "sb_length"].values[
                0] / 1e6
            max_length = max(sb_len_s1,sb_len_s2)

            div_time = get_divergence_time(species1, species2, tree)
            if div_time == 0: continue
            div_time_dict[(species1, species2)] = div_time
            if type == "number":
                count_s1 = tad_sb_df.loc[(tad_sb_df['species'] == species1) & (tad_sb_df['aln'] == aln), "count"].values[0]
                count_s2 = tad_sb_df.loc[(tad_sb_df['species'] == species2) & (tad_sb_df['aln']==aln),"count"].values[0]
                count_per_mb_s1 = count_s1/(sb_len_s1)
                count_per_mb_s2 = count_s2/(sb_len_s2)
                rate_per_mb[(species1, species2)].append(abs(count_per_mb_s1 - count_per_mb_s2))
                count_rate_diff = abs(count_per_mb_s1 - count_per_mb_s2)

            elif type == "borders":
                syn_block = get_equal_tad_count_syn_blocks(tad_sb_df, (species1, species2))
                if aln in syn_block:
                    tads = tad_sb_coord_df.loc[(tad_sb_coord_df["aln"] == aln) &
                                               ((tad_sb_coord_df['species'] == species1) |
                                                (tad_sb_coord_df['species'] == species2))
                                               ]
                    if (len(tads[tads['species']==species1]) >= 3) and (species1 in tads['species'].values) and (species2 in tads['species'].values):
                        count_rate_diff = compute_delta_borders(tads, syn_block, (species1,species2))
                        # r2, pval = map_tad_boundaries(tads, syn_block, (species1,species2))
                        # count_rate_diff = 1 - r2

            elif type == "conformation":
                syn_block = get_equal_tad_count_syn_blocks(tad_sb_df, (species1, species2))
                if aln in syn_block:
                    genes_in_aln = df[aln].dropna()

                    tads = tad_sb_coord_df.loc[(tad_sb_coord_df["aln"] == aln) &
                                               ((tad_sb_coord_df['species'] == species1) |
                                                (tad_sb_coord_df['species'] == species2))
                                               ]
                    if len(tads[tads['species'] == species1]) >= 2 :
                        count_rate_diff,n_genes = calculate_gene_per_tad(species1, species2, genes_in_aln, species_coords, tads, aln,max_length)
                        density[(species1, species2)].append(n_genes/max_length)

            elif type == "edits":
                genes_in_aln = df[aln].dropna()
                tads = tad_sb_coord_df.loc[(tad_sb_coord_df["aln"] == aln) &
                                           ((tad_sb_coord_df['species'] == species1) |
                                            (tad_sb_coord_df['species'] == species2))]

                if ((len(tads[tads['species']==species1]) > 3 and  species2 in tads['species'].values) or
                        (len(tads[tads['species']==species2]) > 3 and  species1 in tads['species'].values)) :
                    count_rate_diff,n_borders = calculate_edit_per_block(species1, species2, genes_in_aln, species_coords, tads, aln,max_length)
                    if n_borders is not None: density[(species1, species2)].append(n_borders/max_length)

            if count_rate_diff is not None: rate_per_mb[(species1, species2)].append(count_rate_diff)

    rate_per_mb = finalize_rates(rate_per_mb, div_time_dict, 'TAD_diff_per_mb',type)
    if type in ["conformation","edits"]:
        return rate_per_mb, div_time_dict,density
    else:
        return rate_per_mb, div_time_dict

def plt_result(rate_per_mb, plot_type, line_length=40):

    x = rate_per_mb['divergence_time']
    y = rate_per_mb['TAD_diff_per_mb']

    categories = {"Mammals": ["#CB1B16",rate_per_mb.loc[rate_per_mb['divergence_time'] < 100]],
                  'Tetrapoda': ["#980E11",rate_per_mb.loc[(rate_per_mb['divergence_time'] >= 100) & (rate_per_mb['divergence_time'] <= 320)]],
                  'Vertebrates': ["#65010C",rate_per_mb.loc[rate_per_mb['divergence_time'] > 320]]}

    plt.figure(figsize=(6.4, 4.8))

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    if type == "number":
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Linear Fit', linewidth=2.5,zorder=1)
        # plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Linear Fit (slope={slope:.2f})', linewidth=2.5)

    else:
        mean_div_times = []
        mean_tad_diffs = []
        added_label = False
        for category, df in categories.items():
            if not df[1].empty:
                mean_div_time = df[1]['divergence_time'].mean()
                mean_tad_diff = df[1]['TAD_diff_per_mb'].median()

                plt.hlines(mean_tad_diff, mean_div_time - line_length / 2, mean_div_time + line_length / 2,
                           color=df[0], linestyle='solid', linewidth=5.5, label=category)
                if not added_label:  added_label = True

                mean_div_times.append(mean_div_time)
                mean_tad_diffs.append(mean_tad_diff)

    plt.scatter(x, y, color='#64b5f6', alpha=0.9,zorder=0,s=100)

    y_labels = {
        'number': 'TAD Number Difference per Mbp',
        'borders': 'TAD Border Difference',
        'conformation': 'TAD Conformation Similarity per Mbp',
        'edits': 'TAD Edits per Mbp'
    }

    plt.xlabel('Divergence Time (Million Years)', fontsize=15)
    plt.ylabel(y_labels.get(plot_type, 'TAD Difference per Mbp'), fontsize=15)

    plt.legend()
    plt.grid(color='gray', linewidth=0.15, alpha=0.7)
    # plt.savefig(f'../images/supplementary/{type}.png', bbox_inches='tight',dpi=600)
    # plt.savefig(f'../images/supplementary/{type}.svg', bbox_inches='tight')
    plt.show()

def main(type):
    aln_code = 'hrmrrcspcdcz'
    coord_path = "/Users/fabianpa/Desktop/new_sequences/coord"
    newick_tree = "(((((((cow:21.62563,sheep:21.62563):40.21702,pig:61.84265):14.15735,(cat:55.36164,dog:54.63914):20.63836):18,(((rat:11.64917,mouse:10.83617):67.32144,rabbit:78.97061):8.22939,(human:27.43052,rhesus:27.49404):58.38):6.8):224.95,chicken:318.95):110.05,zfish:429));"

    tree,species_list, tad_sb_df, tad_sb_coord_df, species_coords = load_species_data(newick_tree,aln_code,coord_path)
    all_aln, species = get_cynteny_df(aln_code)
    reform = {(outerKey, innerKey): values for outerKey, innerDict in all_aln.items() for innerKey, values
              in
              innerDict.items()}
    df = pd.DataFrame.from_dict(reform, orient='index').transpose()
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    column_n = list(range(1, len(df['Alignment1'].columns), 2))
    columns_name = {column_n[i]: species[i] for i in range(len(column_n))}
    df.rename(columns=columns_name, inplace=True)
    sb_coord = get_sb_len(species, aln_code)


    if type in ["conformation","edits"]:
        rate_per_mb, div_time_dict,density = calculate_rates(
            type, tad_sb_df, tad_sb_coord_df, species_list, tree, sb_coord, species_coords, df
        )

    else:
        rate_per_mb, div_time_dict = calculate_rates(
                type, tad_sb_df, tad_sb_coord_df, species_list, tree, sb_coord, species_coords, df
            )


    rate_per_mb['TAD_diff_ratio'] = rate_per_mb['TAD_diff_per_mb'] / rate_per_mb['divergence_time']
    average_ratio = rate_per_mb['TAD_diff_ratio'].mean()

    if type in ["conformation","edits"]:
        rate_per_mb['density']= density

        # rate_per_mb['density_avg'] = rate_per_mb['density'].apply(lambda x: sum(x) / len(x))
        rate_per_mb['density_avg'] = rate_per_mb['density'].apply(lambda x: np.mean(x))
        overall_avg_density = rate_per_mb['density_avg'].mean()
        print(f"Overall Average Gene Density: {overall_avg_density}")

    print(f"Performing evolutionary rates analysis on {type}")
    print(rate_per_mb.sort_values('divergence_time', ascending=True))

    if type == "number":
        print(f"Average rate of TAD change: {average_ratio} TADs per Mb per Myr")

    else:
        categories = {"Mammals": ["#CB1B16", rate_per_mb.loc[rate_per_mb['divergence_time'] < 100]],
                      'Tetrapoda': ["#980E11", rate_per_mb.loc[
                          (rate_per_mb['divergence_time'] >= 100) & (rate_per_mb['divergence_time'] <= 320)]],
                      'Vertebrates': ["#65010C", rate_per_mb.loc[rate_per_mb['divergence_time'] > 320]]}

        for category, df in categories.items():
            if not df[1].empty:
                mean_tad_diff = df[1]['TAD_diff_per_mb'].median()
                if type in ["conformation","edits"]:
                    mean_avg_density = df[1]['density_avg'].median()
                    print(category, round(mean_tad_diff,2),round(mean_avg_density,2))
                else:
                    print(category, round(mean_tad_diff,2))

    plt_result( rate_per_mb,type)

if __name__ == "__main__":
    for type in ["edits"]:
    # for type in ["number","borders","conformation","edits"]:
        main(type)
