import os
import re

from tabulate import tabulate
from tqdm import tqdm

import math
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import linregress
from itertools import product,combinations
from matplotlib.patches import Rectangle


from ete3 import Tree
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_context("paper")

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
    assert isinstance(code, object)
    return alignment_code_to_species.get(code)

def get_sb_len(species, aln, path="synteny_bed"):
    sb_dfs = []
    for common_name in species:
        fname = os.path.join(path, common_name + "_" + aln + ".bed")
        sb_df = pd.read_table(fname, names=("chr", "start", "stop", "aln", "strand"), header=None)
        sb_df["species"] = common_name
        sb_df["sb_length"] = sb_df["stop"] - sb_df["start"]
        sb_dfs.append(sb_df)
    return pd.concat(sb_dfs, axis=0)

def get_tad_in_sb(species, aln, sb_coord, sb_specific_aln=None, shuffle=False,
                  coord_path="sb_in_tad_3/"):
    count_df = []
    coord_df = []
    for common_name in species:
        coord_fname = os.path.join(coord_path, f"{common_name.lower()}_{aln}.tad.sb")
        specie_aln_name = common_name.lower() + "_" + aln
        try:
            sb_tad_coord = \
                pd.read_csv(coord_fname, sep="\t", usecols=[5, 6, 7, 3, 4],
                            names=("aln", "strand", "chr", "start", "stop"),
                            header=None)[["chr", "start", "stop", "aln", "strand"]]
            sb_tad_coord["species"] = common_name
            sb_tad_coord["tad_size"] = sb_tad_coord["stop"] - sb_tad_coord["start"]
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

def get_cynteny_df(aln_name, aln_path='synteny_3/'):
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
            'Gene_name': [i.strip().split()[0] for i in coord_file if len(i.strip().split()) == 5],
            'Chr': [i.strip().split()[1] for i in coord_file if len(i.strip().split()) == 5],
            'Start': [int(i.strip().split()[2]) for i in coord_file if len(i.strip().split()) == 5],
            'End': [int(i.strip().split()[3]) for i in coord_file if len(i.strip().split()) == 5],
            'Strand': [i.strip().split()[4] for i in coord_file if len(i.strip().split()) == 5]}
    coord.close()
    return coord_dict

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

def get_avg_genes_per_tad(tads, alignments, species, gene_coords, aln_number):
    aln = alignments[aln_number].dropna()
    genes = aln[species].str.split('_').str[0]
    if tads.loc[(tads.species == species) & (tads.aln == aln_number)].empty: return 0
    strand = tads.loc[(tads.species == species) & (tads.aln == aln_number), "strand"].values[0]
    tad_s1 = tads.loc[(tads.species == species) & (tads.aln == aln_number)].sort_values(by=['start'], ascending=(
                strand != "-")).reset_index(drop=True)

    # Remove genes with '-' and extract relevant gene coordinates
    genes = genes[genes != '-']
    gene_coords = gene_coords[species]
    gene_coords['Gene_name'] = [item.split('_')[0] for item in gene_coords['Gene_name']]

    gene_distribution = {i: 0 for i in range(len(tad_s1))}

    for i in genes:
        if i in gene_coords['Gene_name']:
            g_index = gene_coords['Gene_name'].index(i)
            g_start, g_stop = gene_coords['Start'][g_index], gene_coords['End'][g_index]

            # Find TAD index that overlaps with gene
            tad_condition = (tad_s1['start'] <= g_start) & (tad_s1['stop'] >= g_stop)
            idx = tad_s1.loc[tad_condition].index[0] if not tad_s1.loc[
                tad_condition].empty else get_gene_overlap_in_tad(g_start, g_stop, tad_s1)

            if idx is not None:
                gene_distribution[idx] += 1

    # Calculate avg number of genes per TAD
    avg_gene_x_tad = sum(gene_distribution.values()) / len(gene_distribution)

    return avg_gene_x_tad

def phylogenetic_covariance_matrix_OU(tree, alpha, sigma):
    """
    Construct the phylogenetic covariance matrix for a OU model.

    Parameters
    ----------
    tree: Phylogenetic tree object.
    alpha
    sigma: variance parameter of the OU model.

    Returns
    -------
    V: Phylogenetic covariance matrix.
    """
    taxa = [leaf.name for leaf in tree.get_leaves()]
    n = len(taxa)
    v = sigma / (2 * alpha)
    # v = sigma ** 2 / (2 * alpha)
    V = np.zeros((n, n))

    for i, leaf_i in enumerate(tree.get_leaves()):
        for j, leaf_j in enumerate(tree.get_leaves()):
            mrca = tree.get_common_ancestor(leaf_i, leaf_j)
            tij= tree.get_distance(leaf_i, leaf_j)
            tra= mrca.get_distance(tree.get_tree_root())
            V[i, j] =  v * np.exp(-alpha * tij) * (1 - np.exp(-2 * alpha * tra))
    return V

def phylogenetic_covariance_matrix_BM(tree, sigma):
    """
    Construct the phylogenetic covariance matrix for a Brownian Motion model.

    Parameters
    ----------
    tree: Phylogenetic tree object.
    sigma: variance parameter of the BM model.

    Returns
    -------
    V: Phylogenetic covariance matrix.
    """

    taxa = [leaf.name for leaf in tree.get_leaves()]
    n = len(taxa)
    V = np.zeros((n, n))

    for i, leaf_i in enumerate(tree.get_leaves()):
        for j, leaf_j in enumerate(tree.get_leaves()):
            mrca = tree.get_common_ancestor(leaf_i, leaf_j)
            if i==j:tij = mrca.get_distance(tree.get_tree_root())
            else: tij= tree.get_distance(leaf_i, leaf_j)
            V[i, j] = V[j, i] = sigma * tij
    return V

def get_design_matrix(tree, alpha):
    n_species = len(tree.get_leaves())
    root_node = tree.get_tree_root()
    C = np.zeros((n_species, 2))

    for i, node in enumerate(tree.get_leaves()):
        parent_node = node.up
        if parent_node is not None:
            tia = parent_node.get_distance(root_node)
        else:
            tia = 0  # For root node or if parent_node is None

        C[i, 0] = np.exp(-alpha * tia)
        C[i, 1] = 1 - np.exp(-alpha * tia)
    return C

def OU_GLS(tree, trait, alpha, sigma):
    """
    Perform Generalized Least Squares estimation under an Ornstein-Uhlenbeck model.

    Parameters
    ----------
    tree: Phylogenetic tree object.
    trait: Array of trait values.
    alpha: reverting mean parameter of the OU model.
    sigma: variance parameter of the OU model.

    Returns
    -------
    theta_hat: Estimated theta parameters.
    log_likelihood: Log-likelihood of the model given the data.
    """

    def compute_log_likelihood(y, X, beta, Sigma):
        n = len(y)
        residual = y - X @ beta
        try:
            log_likelihood = -0.5 * (n * np.log(2 * np.pi) + np.log(np.linalg.det(Sigma)) +
                                     residual.T @ np.linalg.inv(Sigma) @ residual)
        except np.linalg.LinAlgError:
            log_likelihood = -np.inf  # In case of numerical issues with the matrix inversion

        return log_likelihood

    if alpha==0: theta_hat=[0,0]
    else:
        # Construct the phylogenetic covariance matrix Sigma
        V = phylogenetic_covariance_matrix_OU(tree, alpha, sigma)
        try:
            V_inv = inv(V)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix inversion failed, possibly due to numerical instability.")

        # Design matrix C (nx2)
        C = get_design_matrix(tree,alpha)

        # Generalized Least Squares estimation of theta
        try:
            theta_hat = inv(C.T @ V_inv @ C) @ (C.T @ V_inv @ trait)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix inversion in GLS estimation failed, possibly due to numerical instability.")

    log_likelihood = compute_log_likelihood(trait, C, theta_hat, V)
    return  theta_hat,log_likelihood

def grid_search(tree, datasets, alpha_values, sigma_values, aln_codes):
    """
    Perform a grid search to find the best alpha and sigma values for a given phylogenetic tree and dataset.

    Parameters:
    - tree: Phylogenetic tree object.
    - datasets: List of phenotype datasets.
    - alpha_values: List of alpha values to test.
    - sigma_values: List of sigma values to test.
    - aln_codes: List of alignment codes corresponding to the datasets.

    Returns:
    - result: List containing the best parameters for each dataset.
    - cumulative_log_likelihood_matrix: Cumulative log-likelihood matrix for all datasets.
    """

    def calculate_log_likelihood_matrix(tree, phenotypes, alpha_values, sigma_values):
        """
        Calculate the log-likelihood matrix for given phenotypes and parameter values.

        Parameters:
        - tree: Phylogenetic tree object.
        - phenotypes: Phenotype data.
        - alpha_values: List of alpha values.
        - sigma_values: List of sigma values.

        Returns:
        - log_likelihood_matrix: Log-likelihood matrix.
        """
        pair = list(product(alpha_values, sigma_values))
        log_likelihood_matrix = np.zeros((len(alpha_values), len(sigma_values)))

        for j, (alpha, sigma) in enumerate(pair):
            params, loglik = OU_GLS(tree, phenotypes, alpha, sigma)
            log_likelihood_matrix[j // len(sigma_values), j % len(sigma_values)] = round(loglik,3)

        return log_likelihood_matrix

    def find_best_parameters(log_likelihood_matrix, alpha_values, sigma_values):
        """
        Find the best alpha and sigma values from the log-likelihood matrix.

        Parameters:
        - log_likelihood_matrix: Log-likelihood matrix.
        - alpha_values: List of alpha values.
        - sigma_values: List of sigma values.

        Returns:
        - best_alpha: Best alpha value.
        - best_sigma: Best sigma value.
        - best_log_likelihood: Best log-likelihood value.
        """
        best_log_likelihood_idx = np.unravel_index(np.argmax(log_likelihood_matrix), log_likelihood_matrix.shape)
        best_alpha = alpha_values[best_log_likelihood_idx[0]]
        best_sigma = sigma_values[best_log_likelihood_idx[1]]
        best_log_likelihood = log_likelihood_matrix[best_log_likelihood_idx]

        return best_alpha, best_sigma, best_log_likelihood

    result = []
    cumulative_log_likelihood_matrix = np.zeros((len(alpha_values), len(sigma_values)))

    for i, phenotypes in tqdm(enumerate(datasets), desc="SB analysed:",colour="magenta"):
        if len(set(phenotypes)) != 1:
            log_likelihood_matrix = calculate_log_likelihood_matrix(tree, phenotypes, alpha_values, sigma_values)
            cumulative_log_likelihood_matrix += log_likelihood_matrix

            best_alpha, best_sigma, best_log_likelihood = find_best_parameters(log_likelihood_matrix, alpha_values,
                                                                               sigma_values)

            best_y0, best_thetas = np.round(OU_GLS(tree, phenotypes, best_alpha, best_sigma)[0], 2)

            result.append([aln_codes[i], best_alpha, best_sigma, best_thetas, best_log_likelihood])

    return result, cumulative_log_likelihood_matrix

def plot_heatmap_loglik(result,best_alpha,best_sigma):
    sns.set_style("white")

    f, ax = plt.subplots(figsize=(19, 16))
    df = result.groupby(['alpha','sigma'])["likelihood"].sum().reset_index()
    heatmap_df = df.pivot(index="alpha", columns="sigma", values="likelihood")
    best_result = df.loc[df['likelihood'].idxmax()]
    idx_alpha = heatmap_df.index.get_loc(best_alpha)
    idx_sigma = heatmap_df.columns.get_loc(best_sigma)
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='coolwarm',square=True,cbar_kws={"shrink": .85}, annot_kws={"size": 15},ax=ax)
    ax.add_patch( Rectangle((idx_sigma, idx_alpha), 1, 1, fill=False, edgecolor='black', linewidth=2.5))

    plt.xlabel('Sigma',fontsize=25)
    plt.ylabel('Alpha',fontsize=25)
    plt.title('Summed Likelihood Heatmap',fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show()

def plot_heatmap_alpha_sigma(result):
    sns.set_style("white")
    f, ax = plt.subplots(figsize=(19, 16))
    counts_df = result.groupby(['alpha', 'sigma']).size().reset_index(name='count')

    best_result = counts_df.loc[counts_df['count'].idxmax()]
    best_alpha = best_result['alpha']
    best_sigma = best_result['sigma']
    print(f"Best Alpha: {best_alpha}")
    print(f"Best Sigma: {best_sigma}")

    heatmap_df = counts_df.pivot(index='alpha', columns='sigma', values='count')
    idx_alpha = heatmap_df.index.get_loc(best_alpha)
    idx_sigma = heatmap_df.columns.get_loc(best_sigma)

    ax.tick_params(length=0)
    ax = sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='coolwarm',ax=ax,square=True,cbar_kws={"shrink": .75}, annot_kws={"size": 15})
    ax.add_patch( Rectangle((idx_sigma, idx_alpha), 1, 1, fill=False, edgecolor='black', linewidth=2.5))
    plt.xlabel('Sigma',fontsize=25)
    plt.ylabel('Alpha',fontsize=25)
    plt.title('Heatmap of counts for alpha, sigma pairs',fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show()
    return best_alpha,best_sigma

def plot_cum_loglik_heatmap(matrix, alpha_values, sigma_values):
    f, ax = plt.subplots(figsize=(19, 16))
    print(tabulate(matrix, headers='keys', tablefmt='psql'))

    heatmap = sns.heatmap(matrix, xticklabels=sigma_values, yticklabels=alpha_values, annot=False,vmin = -20000, vmax = -6000, square=True,cmap='coolwarm', fmt='%d',
                          ax=ax,cbar_kws={"shrink": .75,'label': 'Cumulative loglik'},annot_kws={"size": 15})

    plt.title("Cumulative Log Likelihood Heatmap",fontsize=30)
    plt.xlabel("Sigma Values",fontsize=30)
    plt.ylabel("Alpha Values",fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=25)

    plt.show()

def get_slope(result,tad_sb_df,tree):
    def lin_func(x, a, b):
        y = a * x + b
        return y
    species = [node.name for node in tree.get_leaves()]
    species_pairs = list(combinations(species, 2))
    slopes = {}
    for i in range(len(result)):
        code = result[i][0]
        differences = []
        count = tad_sb_df.loc[tad_sb_df.aln == code, ["count", "species"]]
        for s1, s2 in species_pairs:
            difference = abs(
                count.loc[count.species == s1, "count"].values[0] - count.loc[count.species == s2, "count"].values[0])
            node1 = tree.search_nodes(name=s1)[0]
            node2 = tree.search_nodes(name=s2)[0]
            distance = node1.get_distance(node2)
            differences.append([(s1, s2), distance, difference])
        df = pd.DataFrame(differences, columns=["specie_pair", "evolutionary_distance", "difference"])
        df = df.loc[df["difference"] > 0]

        # Log-transform the data
        df["log_evolutionary_distance"] = np.log(df["evolutionary_distance"])
        df["log_difference"] = np.log(df["difference"])
        slope, intercept, r_value, p_value, std_err = linregress(df["log_evolutionary_distance"],
                                                                 df["log_difference"])
        slopes[code]=slope
    slopes_df = pd.DataFrame.from_dict(slopes,orient="index",columns=["slope"]).reset_index(names="aln")
    return slopes_df

def plot_couple_values(result,tad_sb_df,tree):

    def lin_func(x, a, b):
        y = a * x + b
        return y

    custom_palette = ["#ff595e", "#ffca3a", "#8ac926", "#6a4c93", "#1982c4", "#313dbf", "#fe218b", "#ec0700", "#f6a1c0",
                      "#b2f7ef", "#bf10d8",
                      "#ff6000", "#00cfc1", "#000edd", "#9d0208", "#161a1d", "#a5ffd6", "#60d394", "#6699cc", "#fff275",
                      "#ff3c38", "#254441",
                      "#a23e48", "#18206f", "#ff9000", "#ffcbdd", "#c5d86d", "#bd4ef9", "#ed0101", "#fee719", "#19a5ff",
                      "#fe7295", "#7ddf64",
                      "#56494e", "#511c29", "#f4845f", "#d8e2dc", "#004e89", "#73a580", "#f21b3f", "#ff9914", "#b4436c",
                      "#1b512d", "#2bd9fe",
                      "#f2e2ba", "#bad7f2", "#5a189a", "#2176ff", "#e9190f", "#688e26", "#ffcc00", "#9683ec", "#2a9d8f",
                      "#023047", "#e76f51",
                      "#ff006e", "#833907", "#e9ff70", "#bbdbfe", "#f58f29", "#5438dc", "#d138bf", "#204b57", "#d7263d",
                      "#090c9b", "#8fc93a"]

    num_plots = len(result)
    species = [node.name for node in tree.get_leaves()]
    species_pairs = list(combinations(species, 2))
    ncols = math.ceil(math.sqrt(num_plots))
    nrows = math.ceil(num_plots / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    for i, ax in enumerate(axs.flatten()):
        if i < num_plots:
            code = result[i][0]

            differences = []
            count = tad_sb_df.loc[tad_sb_df.aln == code, ["count", "species"]]
            for s1,s2 in species_pairs:
                difference = abs(count.loc[count.species == s1, "count"].values[0] - count.loc[count.species == s2, "count"].values[0])
                node1 = tree.search_nodes(name=s1)[0]
                node2 = tree.search_nodes(name=s2)[0]
                distance = node1.get_distance(node2)
                differences.append([(s1,s2),distance,difference])

            custom_palette = custom_palette[:len(species_pairs)]
            color_species_correspondence = {species_pairs[i]: color for i, color in enumerate(custom_palette)}
            df = pd.DataFrame(differences, columns=["specie_pair", "evolutionary_distance", "difference"])
            df = df.loc[df["difference"] > 0]
            df["color"] = df["specie_pair"].map(color_species_correspondence)

            # Log-transform the data
            df["log_evolutionary_distance"] = np.log(df["evolutionary_distance"])
            df["log_difference"] = np.log(df["difference"])
            slope, intercept, r_value, p_value, std_err = linregress(df["log_evolutionary_distance"],
                                                                     df["log_difference"])

            sns.scatterplot(data=df,x="log_evolutionary_distance",y= "log_difference",hue="specie_pair", ax=ax,palette=df["color"].tolist(),legend=False)
            sns.lineplot(x=df["log_evolutionary_distance"], y=intercept + slope * df["log_evolutionary_distance"],ax=ax,color="r", linestyle="dashed")

            ax.set_xlabel('Phylogenetic Distance (log)')
            ax.set_ylabel('Trait Difference (log)')
            ax.set_title(f"SB: {code.split('_')[1]}  Slope:{round(slope,2)}")

        else:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

def plot_half_life(result_df,alpha_values):
    sns.set_style("white")
    plt.figure(figsize=(6.4, 4.8))
    #Plot half life
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(wspace=0.09)  # Adjust space between plots
    sns.histplot(result_df.loc[result_df['half_life']<10,'half_life'],bins=10, kde=False, color='skyblue', edgecolor='black',ax=ax1)
    sns.histplot(result_df.loc[result_df['half_life']>60,'half_life'],bins=1, kde=False, color='skyblue', edgecolor='black',ax=ax2)
    ax1.set_xlim(0, 10)
    ax2.set_xlim(65, 71)

    # Hide the spines between the two plots
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)

    # Add slanted lines to indicate the break
    d = .015  # How big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1.5)
    ax1.plot((.99 , 1 ), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)  # switch to the right axes
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    # Add axis labels
    ax1.set_xlabel('                            Half Life',fontsize=15)
    ax2.set_xlabel('')
    ax1.set_ylabel('Frequency',fontsize=15)

    plt.show()

def plot_half_life_bar(result_df, alpha_values):
    sns.set_style("white")
    plt.figure(figsize=(8, 6))

    plot_data = []

    for alpha in alpha_values:
        counts = result_df['half_life'].value_counts().reset_index()
        counts.columns = ['half_life', 'count']
        counts['alpha'] = alpha

        plot_data.append(counts)

    plot_df = pd.concat(plot_data)
    sns.barplot(x='half_life', y='count', data=plot_df)

    plt.xlabel('Half Life', fontsize=15)
    plt.ylabel('Count', fontsize=15)

    plt.show()

def plot_alpha_bar(result_df, alpha_values):
    sns.set_style("white")
    plt.figure(figsize=(8, 6))

    plot_data = []

    counts = result_df['alpha'].value_counts().reset_index()

    # plot_df = pd.concat(plot_data)
    sns.barplot(x='alpha', y='count', data=counts)
    plt.xlabel('Alpha Values', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.savefig("alpha_bar.png", bbox_inches='tight', dpi=600)
    plt.show()

def plot_correlation(result_df):
    sns.displot(result_df, x="theta", kind="kde")
    # plt.savefig("half_life.png",bbox_inches='tight')
    # plt.savefig("half_life.svg",bbox_inches='tight')
    plt.show()

def process_tad_genes(aln_number,aln="hrmrrcspcd"):
    coord_path = "/Users/fabianpa/Desktop/new_sequences/coord"
    all_aln, species = get_cynteny_df(aln)
    reform = {(outerKey, innerKey): values for outerKey, innerDict in all_aln.items() for innerKey, values in
              innerDict.items()}
    df = pd.DataFrame.from_dict(reform, orient='index').transpose()
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    column_n = list(range(1, len(df['Alignment1'].columns), 2))
    columns_name = {column_n[i]: species[i] for i in range(len(column_n))}
    df.rename(columns=columns_name, inplace=True)
    sb_coord = get_sb_len(species, aln)
    species_coords = {f: read_coord_file(os.path.join(coord_path, f + ".coord")) for f in species}
    tad_sb_list, tad_sb_coord_list = get_tad_in_sb(species, aln, sb_coord)
    tad_sb_df = pd.concat(tad_sb_list, axis=0)
    tad_sb_coord_df = pd.concat(tad_sb_coord_list, axis=0)
    tad_sb_df.sort_values(by='aln', inplace=True)
    avg_gene_x_tad_df = []
    for i in species:
        comb_df = tad_sb_coord_df.loc[(tad_sb_coord_df.species == i)]
        avg_genes_per_tad = get_avg_genes_per_tad(comb_df, df, i, species_coords, aln_number)
        avg_gene_x_tad_df.append([avg_genes_per_tad, i])
    result = pd.DataFrame(avg_gene_x_tad_df, columns=["count", "species"])
    return result

def get_genes_in_aln(aln_number,aln="hrmrrcspcd"):
    all_aln, species = get_cynteny_df(aln)
    reform = {(outerKey, innerKey): values for outerKey, innerDict in all_aln.items() for innerKey, values in
              innerDict.items()}
    df = pd.DataFrame.from_dict(reform, orient='index').transpose()
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    column_n = list(range(1, len(df['Alignment1'].columns), 2))
    columns_name = {column_n[i]: species[i] for i in range(len(column_n))}
    df.rename(columns=columns_name, inplace=True)
    df_filtered = df[aln_number]
    return df_filtered

def block_stat(aln_number,sb_coord,tad_sb_df):
    genes = get_genes_in_aln(aln_number)

    long_life = genes.xs("human", level=1, axis=1).copy()
    long_life[aln_number] = long_life[aln_number].replace('-', pd.NA)
    mean_genes = [ ]
    mean_count_per_mb = []
    avg_tad_n = []
    avg_tad_n_per_mb = []

    for i in aln_number:
        n_genes = len(long_life[i].dropna())
        mean_genes.append(n_genes)
        sb_len = sb_coord.loc[(sb_coord['species'] == "human") & (sb_coord['aln'] == i), "sb_length"].values[
                     0] / 1e6
        tad_n = tad_sb_df.loc[(tad_sb_df.aln == i) & (tad_sb_df.species == "human"), ["count"]].values[0]
        count_per_mb = n_genes/sb_len
        mean_count_per_mb.append(count_per_mb)
        avg_tad_n.append(tad_n)
        avg_tad_n_per_mb.append(tad_n/sb_len)

    return  len(aln_number),round(np.median(mean_genes),2),round(np.median(mean_count_per_mb),2),round(np.median(avg_tad_n),2),round(np.median(avg_tad_n_per_mb),2)

def get_go_input(genes,out="high_half_life_genes.csv"):
    genes = genes.xs("human", level=1, axis=1).values.flatten()

    filtered_genes = [x.split("_")[0] for x in genes if not pd.isna(x) and x!='-']

    # print(filtered_genes)
    # with open(out, 'w') as output:
    #     for item in filtered_genes:
    #         output.write("{}\n".format(item))

def main(species_aln,mode="count"):

    tree = Tree("mammals_tree_mya.nw", format=1)
    species_nodes = tree.get_leaf_names()

    species = get_code_conversion(species_aln).split(";")
    sb_coord = get_sb_len(species, species_aln)

    tad_sb_list, tad_sb_coord_list = get_tad_in_sb(species, species_aln, sb_coord)
    tad_sb_df = pd.concat(tad_sb_list, axis=0)
    tad_sb_df.sort_values(by='aln', inplace=True)


    datasets=[]

    aln_codes=tad_sb_df.aln.unique()[:]
    for aln in tqdm(aln_codes,"Getting average gene per TAD....", colour="blue"):

        if mode == "count": count = tad_sb_df.loc[tad_sb_df.aln == aln, ["count", "species"]]
        else: count = process_tad_genes(aln)

        # Check that the order of the species is the same as the species in the tree's leaf names
        count['species'] = pd.Categorical(count['species'], categories=species_nodes, ordered=True)
        count_sorted = count.sort_values(by='species')
        datasets.append(count_sorted["count"].values)

    alpha_values = [ 0.01,  0.09,  0.17,  0.26,  0.34,  0.67,  1.,  2.,  3.6,  5.2,  6.8,  8.4,  10.]
    sigma_values = [ 0.01,  0.54, 1.06, 1.59,  2.11,  2.64, 3.16,  3.69,  4.22,  4.74,  5.27, 5.79,
                     6.32,  6.84,  7.37,  7.90,  8.42,  8.95, 9.47,10.]

    result,cumulative_log_likelihood_matrix = grid_search(tree, datasets, alpha_values,sigma_values,aln_codes)
    result_df = pd.DataFrame(result,columns=["aln","alpha","sigma","theta","likelihood"])

    # Calculate half-life and stationary variance
    result_df['half_life'] = round(np.log(2) / result_df['alpha'],2)
    result_df['stationary_variance'] = result_df['sigma'] ** 2 / (2 * result_df['alpha'])

    #Plot Results
    plot_alpha_bar(result_df,alpha_values)
    plot_half_life_bar(result_df,alpha_values)
    plot_correlation(result_df)

    plot_couple_values(result,tad_sb_df,tree)
    best_alpha,best_sigma = plot_heatmap_alpha_sigma(result_df)
    plot_heatmap_loglik(result_df,best_alpha,best_sigma)
    plot_cum_loglik_heatmap(cumulative_log_likelihood_matrix, alpha_values, sigma_values)

    # Get block statistic for each half life
    half_life_statistic = []
    for half_life in  result_df['alpha'].unique():
        aln_number = result_df.loc[result_df['alpha'] == half_life, "aln"].tolist()

        n_blocks,mean_n_genes,count_per_mb,avg_tad_n,avg_tad_n_per_mb = block_stat(aln_number,sb_coord,tad_sb_df)
        half_life_statistic.append([half_life,n_blocks,mean_n_genes,count_per_mb,avg_tad_n,avg_tad_n_per_mb])

    half_life_statistic_df = pd.DataFrame(half_life_statistic,columns=["half_life","n_blocks","avg_n_genes","avg_ngenes_mb","avg_tad_n","avg_tad_n_per_mb"]).sort_values(by='half_life', ascending=True)
    print(half_life_statistic_df)
    half_life_statistic_df.to_csv("../images/supplementary/ou_block_stat.1.csv")
    # get_go_input(genes, out="high_half_life_genes.csv")

if __name__ == "__main__":
    aln = "hrmrrcspcd"
    main(aln,'count')
    # main(aln,'gene')
