import os
import re
import time
import random

from itertools import combinations
import pandas as pd
import numpy as np
import multiprocessing

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

sns.set_style("whitegrid")
sns.set_context("paper")

div_time = {'mr': 13.10, 'cs': 24.60,'hr': 28.82, 'cd': 55.4, 'csp': 61.8,'cspcd': 76.0,'mrr': 79.0,  'hrmrr': 87.2,
            'hrmrrcspcd': 94.0, 'hrmrrcspcdc': 318.9, 'hrmrrcspcdcz': 429.0}

species_mapping = {'hr':'primates','mr':'rodents','cs':'cattle','cd':'carnivores','csp':'ungulates','cspcd':'laurasians','mrr':'glires','hrmrr':'eurarchontoglires','hrmrrcspcd':'mammals','hrmrrcspcdc':'tetrapods',"hrmrrcspcdcz":"vertebrates"}
colors = {'rodents': "#033270",
          'cattle': "#1368AA",
          'primates': "#4091C9",
          'carnivores': "#4AA8CD",
          'ungulates': "#BA746B",
          'laurasians': "#F29479",
          'glires': "#F26A4F",
          'eurarchontoglires': "#EF3C2D",
          'mammals': "#CB1B16",
          'tetrapods': "#980E11",
          'vertebrates': "#65010C"
          }
colors_sorted = {i:colors[i] for i in colors.keys() }

def get_code_conversion(code):
    alignment_code_to_species = {'hr': 'human;rhesus',
                                 'mr': 'mouse;rat',
                                 'cd': 'cat;dog',
                                 'cs': 'cow;sheep',
                                 'csp': 'pig;cow;sheep',
                                 'mrr': 'mouse;rabbit;rat',
                                 'cspcd': 'pig;cow;sheep;dog;cat',
                                 'hrmrr': 'human;mouse;rhesus;rabbit;rat',
                                 'hrmrrcspcd': 'human;mouse;rhesus;rabbit;rat;pig;cow;sheep;dog;cat',
                                 'hrmrrcspcdc': 'human;mouse;rhesus;rabbit;rat;pig;cow;sheep;dog;cat;chicken',
                                 'hrmrrcspcdcz': 'human;mouse;rhesus;rabbit;rat;pig;cow;sheep;dog;cat;chicken;zfish'}
    return alignment_code_to_species.get(code)

def read_coord_file(fname):
    with open(fname, 'r') as coord:
        coord_file = coord.readlines()[1:]
        coord_dict = {
            'Gene_name': [i.strip().split()[0].split('_')[0] for i in coord_file if len(i.strip().split()) == 5],
            'Chr': [i.strip().split()[1] for i in coord_file if len(i.strip().split()) == 5],
            'Start': [int(i.strip().split()[2]) for i in coord_file if len(i.strip().split()) == 5],
            'End': [int(i.strip().split()[3]) for i in coord_file if len(i.strip().split()) == 5],
            'Strand': [i.strip().split()[4] for i in coord_file if len(i.strip().split()) == 5]}
    coord.close()
    return coord_dict

def get_cynteny_df(aln_name, aln_path='synteny/'):
    aln_fname = os.path.join(aln_path, aln_name + ".synteny")
    with open(aln_fname, 'r') as synteny_file:
        synteny = synteny_file.read()
        species = [s.split(".coord")[0].split('/')[-1] for s in synteny.split('\n', 1)[0].split()[1:]]
        found_aln = re.findall(r"Alignment*", synteny, re.MULTILINE)
        n_aln = list(map(lambda x, y: x + str(y), found_aln, range(1, len(found_aln) + 1)))
        all_aln = {key: {} for key in n_aln}
        len_row = len(synteny.split('Ali')[1].split('\n')[1].split())
        n_keys = list(range(1,len_row,2))
        for i in synteny.split('Ali'):
            aln = None
            for j in i.split('\n'):
                if j.startswith('gnment '):
                    aln = 'Alignment' + j.split()[1]
                    all_aln[aln] = {key: [] for key in n_keys}
                elif aln:
                    [all_aln[aln][idx].append(j.strip().split()[idx].split('_')[0]) for idx in
                     range(len(j.strip().split())) if idx%2!=0]
    return all_aln, species

def get_sb_len(species,aln,path="/Users/fabianpa/Desktop/new_sequences/synteny_3/synteny_bed"):
    sb_dfs = []
    for common_name in species:
        fname = os.path.join(path,common_name+"_"+aln+".bed")
        sb_df = pd.read_table(fname, names=("chr", "start", "stop", "aln", "strand"),header=None)
        sb_df["species"] = common_name
        sb_df["sb_length"]  = sb_df["stop"]- sb_df["start"]
        sb_dfs.append(sb_df)
    return pd.concat(sb_dfs, axis=0)

def get_tad_in_sb(species, aln, coord_path="sb_in_tad/"):
    count_df = []
    coord_df = []
    for common_name in species:
        coord_fname = os.path.join(coord_path, f"{common_name.lower()}_{aln}.tad.sb")
        try:
            sb_tad_coord = \
                pd.read_csv(coord_fname, sep="\t", usecols=[5, 6, 7, 3, 4],
                            names=("aln", "strand", "chr", "start", "stop"),
                            header=None)[["chr", "start", "stop", "aln", "strand"]]
            sb_tad_coord["species"] = common_name
            sb_tad_coord["tad_size"] = sb_tad_coord["stop"] - sb_tad_coord["start"]
            coord_df.append(sb_tad_coord)
            tad_count = sb_tad_coord['aln'].value_counts().sort_index().to_frame()
            tad_count['species'] = common_name
            count_df.append(tad_count)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e.filename}")
    return count_df, coord_df

def get_gene_overlap_in_tad(g_start, g_stop, tads):
    tad_modified = tads.copy()
    tad_modified.start = tad_modified.start - 40000
    tad_modified.stop = tad_modified.stop + 40000
    strand = tad_modified['strand'].values[0]
    if strand == '-':
        closest_tad = tad_modified.iloc[(tad_modified['stop'] - g_start).abs().argsort()[0]]
    else:
        closest_tad = tad_modified.iloc[(tad_modified['start'] - g_start).abs().argsort()[0]]
    idx = closest_tad.name
    if (closest_tad['start'] <= g_start) and (closest_tad['stop'] >= g_stop): return idx

def get_synonyms(syn_path, g1, g2):
    matching_gene = []
    synonyms_1 = pd.read_csv(syn_path[0], sep='\t').assign(Symbol=lambda df: df["Symbol"].str.upper(),
                                                           Synonyms=lambda df: df["Synonyms"].str.upper())
    synonyms_2 = pd.read_csv(syn_path[1], sep='\t').assign(Symbol=lambda df: df["Symbol"].str.upper(),
                                                           Synonyms=lambda df: df["Synonyms"].str.upper())
    syn_g1 = synonyms_1.loc[synonyms_1["Symbol"].isin(g1), ["Symbol", "Synonyms"]].dropna().drop_duplicates()
    syn_g2 = synonyms_2.loc[synonyms_2["Symbol"].isin(g2), ["Symbol", "Synonyms"]].dropna().drop_duplicates()
    syn_set_1 = set(",".join(syn_g1.Synonyms.values).split(","))
    syn_set_2 = set(",".join(syn_g2.Synonyms.values).split(","))
    if g1.intersection(syn_set_2): matching_gene.extend(list(g1.intersection(syn_set_2)))
    # elif g2.intersection(syn_set_1):
    if not syn_g1[~syn_g1.Symbol.isin(g1.intersection(syn_set_2))].empty:
        syn_g1 = syn_g1[~syn_g1.Symbol.isin(g1.intersection(syn_set_2))]
        syn_g2 = syn_g2[~syn_g2.Symbol.isin(g2.intersection(syn_set_1))]
        syn_set_1 = set(",".join(syn_g1.Synonyms.values).split(","))
        syn_set_2 = set(",".join(syn_g2.Synonyms.values).split(","))
        intersection_1 = syn_set_1.intersection(syn_set_2)
        pattern = '|'.join(intersection_1)
        matching_gene.extend(syn_g1.loc[syn_g1["Symbol"].str.contains(pattern), "Symbol"].values)
    return matching_gene

def get_tad_number(gene_inters, tad_s1, tad_s2, gene_coords, species, mode="inters"):
    gene_distribution = {0: {i: [] for i in range(len(tad_s1))},
                         1: {i: [] for i in range(len(tad_s2))}}

    gene_coords_0 = gene_coords[species[0]]
    gene_coords_1 = gene_coords[species[1]]

    for i in gene_inters:
        if mode == "inters" and i != "-":
            g1, g2 = i, i
            g1_index, g2_index = gene_coords_0['Gene_name'].index(g1), gene_coords_1['Gene_name'].index(g2)

            g1_start, g1_stop = gene_coords_0['Start'][g1_index], gene_coords_0['End'][g1_index]
            g2_start, g2_stop = gene_coords_1['Start'][g2_index], gene_coords_1['End'][g2_index]

            idx_1 = tad_s1.loc[(tad_s1['start'] <= g1_start) & (tad_s1['stop'] >= g1_stop)].index[0] if not \
                tad_s1.loc[
                    (tad_s1['start'] <= g1_start) & (tad_s1['stop'] >= g1_stop)].empty else get_gene_overlap_in_tad(
                g1_start, g1_stop, tad_s1)

            idx_2 = tad_s2.loc[(tad_s2['start'] <= g2_start) & (tad_s2['stop'] >= g2_stop)].index[0] if not \
                tad_s2.loc[
                    (tad_s2['start'] <= g2_start) & (tad_s2['stop'] >= g2_stop)].empty else get_gene_overlap_in_tad(
                g2_start, g2_stop, tad_s2)

            if idx_1 is not None and idx_2 is not None:
                gene_distribution[0][idx_1].append(g1)
                gene_distribution[1][idx_2].append(g2)

    return gene_distribution

def calculate_edit_dist(alignments, tads, species, gene_coords, shuffle=False):
    edit_count = []

    for i, alignment_number in enumerate(tads.aln.unique()[:]):
        n_edit = 0
        aln = alignments[alignment_number].dropna()

        gene_s1 = aln[species[0]]
        gene_s2 = aln[species[1]]
        tad_data_s1 = tads[(tads.species == species[0]) & (tads.aln == alignment_number)]
        tad_data_s2 = tads[(tads.species == species[1]) & (tads.aln == alignment_number)]

        if not tad_data_s1.empty and not tad_data_s2.empty:
            strand_1 = tad_data_s1.iloc[0]['strand']
            strand_2 = tad_data_s2.iloc[0]['strand']

            tad_s1 = tad_data_s1.sort_values(by=['start'], ascending=(strand_1 != "-")).reset_index(drop=True)
            tad_s2 = tad_data_s2.sort_values(by=['start'], ascending=(strand_2 != "-")).reset_index(drop=True)

            intersection = set(gene_s1).intersection(set(gene_s2))
            if intersection:
                gene_distributions = get_tad_number(intersection, tad_s1, tad_s2, gene_coords, species)
                value_to_key_mapping = {value: key for key, values in gene_distributions[1].items() for value in values}

                if shuffle:
                    all_values = [value for values in gene_distributions[0].values() for value in values]
                    random.shuffle(all_values)
                    shuffled_dict = {}
                    values_index = 0
                    for key in gene_distributions[0]:
                        num_values = len(gene_distributions[0][key])
                        shuffled_dict[key] = all_values[values_index:values_index + num_values]
                        values_index += num_values
                    result = {key: set([value_to_key_mapping[value] for value in values]) for key, values in
                              shuffled_dict.items()}
                else:
                    result = {key: set([value_to_key_mapping[value] for value in values]) for key, values in
                              gene_distributions[0].items()}

                insertions = sum(len(v) - 1 for k, v in result.items() if v)
                deletions = sum(
                    len(value1 & value2) for key1, value1 in result.items() for key2, value2 in result.items() if
                    key1 < key2 and len(value1 & value2) > 0)
                n_edit += (deletions + insertions)
                edit_count.append(n_edit)

    return np.mean(edit_count) if edit_count else None

def process_alignment(aln, shuffle=False):
    coord_path = "coord"

    all_aln, species = get_cynteny_df(aln)
    reform = {(outerKey, innerKey): values for outerKey, innerDict in all_aln.items() for innerKey, values in
              innerDict.items()}
    df = pd.DataFrame.from_dict(reform, orient='index').transpose()
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    first = list(all_aln.keys())[0]
    column_n = df[first].columns.values
    columns_name = {column_n[i]: species[i] for i in range(len(column_n))}
    df.rename(columns=columns_name, inplace=True)

    species_coords = {f: read_coord_file(os.path.join(coord_path, f + ".coord")) for f in species}

    tad_sb_list, tad_sb_coord_list = get_tad_in_sb(species, aln)
    tad_sb_df = pd.concat(tad_sb_list, axis=0)
    tad_sb_coord_df = pd.concat(tad_sb_coord_list, axis=0)
    tad_sb_df.sort_values(by='aln', inplace=True)

    comb = list(combinations(species, 2))
    edit_ops = []
    edit_ops_rdm = []
    for i in comb:
        if shuffle:
            for _ in range(1000):
                n_edit_combo_rdm = calculate_edit_dist(df, tad_sb_coord_df, i, species_coords,
                                                       shuffle=True)
                if n_edit_combo_rdm is not None: edit_ops_rdm.append(n_edit_combo_rdm)
        else:
            n_edit_combo = calculate_edit_dist(df, tad_sb_coord_df, i, species_coords)
            if n_edit_combo is not None: edit_ops.append(n_edit_combo)

    if shuffle:
        return [aln, div_time[aln], np.mean(edit_ops_rdm),edit_ops_rdm]
    else:
        return [aln, div_time[aln], np.mean(edit_ops)]

def plot_scatter_fitted(edit_operations_df):

    plt.figure(figsize=(15, 11))

    plt.axhline(y=1, color='r', linestyle="--", label='Reference Line')

    # Scatter plot for data points
    sns.scatterplot(x=edit_operations_df["mya"], y=edit_operations_df["ratio"], hue=edit_operations_df.alignment,
                    palette=list(colors_sorted.values()), s=90)

    # Add labels and legend
    plt.xlabel("Time (Million Years Ago)", fontsize=18)
    plt.ylabel("Gene-TAD vs. Gene-Shuffled TAD Edit Distance Ratio", fontsize=18)

    plt.show()

def plot_pval_distr(merged_df):
    ncols = 3
    nrows = (len(merged_df.alignment) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    axes = axes.flatten()

    for idx, (ax, (_, row)) in enumerate(zip(axes, merged_df.iterrows())):
        alignment = row["alignment"]
        mmr_distr = row["edit_ops_distr"]
        mmr_actual = row["edit_ops_actual"]
        # sns.histplot(mmr_distr, color="skyblue", label="Edit Distance Random Distribution", ax=ax)
        sns.kdeplot(mmr_distr, color="skyblue", label="Edit Distance Random Distribution", ax=ax,)
        ax.axvline(x=mmr_actual, color="red", linestyle="--", label=f"Edit Distance Actual = {round(mmr_actual, 2)}")
        ax.set_title(alignment)
        ax.legend(fontsize=15)
        ax.set_xlabel("Random Edit Distance Values", fontsize=15)
        ax.set_ylabel("Frequency", fontsize=15)
        # ax.set_xlim(0, 1)

    # Hide empty subplots
    for ax in axes[len(merged_df):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("edit_dist_pval.svg", bbox_inches='tight')
    plt.savefig("edit_dist_pval.png", bbox_inches='tight', dpi=600)
    plt.show()

def process_alignment_wrapper(aln):
    print(f"Processing Alignments - {aln}")
    result = process_alignment(aln)
    return result

def process_rdm_alignment_wrapper(aln):
    print(f"Processing Random Alignments - {aln}")
    rdm_result = process_alignment(aln,True)
    return rdm_result

def main():

    start_time = time.time()

    pool = multiprocessing.Pool()

    results = pool.map(process_alignment_wrapper, div_time.keys())
    rdm_results = pool.map(process_rdm_alignment_wrapper, div_time.keys())

    pool.close()
    pool.join()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    edit_operations_df = pd.DataFrame(results, columns=['alignment', 'mya', 'edit_ops'])
    rdm_edit_operations_df = pd.DataFrame(rdm_results, columns=['alignment', 'mya', 'edit_ops','edit_ops_distr'])

    merged_df = edit_operations_df.merge(rdm_edit_operations_df, on=['alignment', 'mya'],
                                          suffixes=('_actual', '_random'))
    merged_df['ratio'] = merged_df['edit_ops_actual'] / merged_df['edit_ops_random']
    merged_df["alignment"] = merged_df["alignment"].replace(species_mapping)

    def filter_list(row):
        if row['alignment'] == 'tetrapods':
            return [value for value in row['edit_ops_distr'] if value > 9]
        elif row['alignment'] == 'vertebrates':
            return [value for value in row['edit_ops_distr'] if value > 5]
        else:
            return row['edit_ops_distr']

    merged_df['edit_ops_distr'] = merged_df.apply(filter_list, axis=1)

    X = merged_df[['alignment','edit_ops_actual','edit_ops_random', 'ratio']]
    print(tabulate(X, headers='keys', tablefmt='psql'))

    plot_pval_distr(merged_df)
    # plot_scatter_fitted(merged_df)

if __name__ == "__main__":
    main()
