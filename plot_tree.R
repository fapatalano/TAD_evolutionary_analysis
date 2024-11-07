library("ape")
library(rphylopic)
library("ggtree")

tree <- read.tree("/Users/fabianpa/Desktop/phdProject/paper_script/vertebrate_tree_mya.nwk")
phylopic_info <- data.frame(
  node = c(1,2,3,4,5,6,7,8,9,10,11,12),
  common_name = c("Cow","Sheep","Pig","Cat","Dog","Rat","Mouse","Rabbit","Human","Rhesus","Chicken","Zebrafish"),
  phylopic = c("ac286b37-bd95-4f39-8d81-756ec348beb9",
               "fbe25684-0b48-4569-947b-2702822cf5b0",
               "008d6d88-d1be-470a-8c70-73625c3fb4fb",
               "23cd6aa4-9587-4a2e-8e26-de42885004c9",
               "1a9332fc-aeac-43ae-a318-80acbf89b662",
               "828a8d15-6aa9-41ab-85a3-e9e06c0f1945",
               "c4572239-3b7c-4259-8049-3f44e6d52e6f",
               "ad7d2a9d-ef0d-46f2-895d-7c67bb8e6355",
               "acf1cbec-f6ef-4a82-8ab5-be2f963b93f5",
               "eedde61f-3402-4f7c-9350-49b74f5e1dba",
               "aff847b0-ecbd-4d41-98ce-665921a6d96e",
               "e86bc377-3a6c-4efa-8703-1a12cb019ef7")
)
#
# '#6e1423' -
# '#641220' - ZFISH
#
color <- c('#e01e37', '#bd1f36', '#a71e34', '#a11d33', '#c71f37', '#6e1423', '#85182a', '#641220', '#da1e37', '#b21e35', '#E12A40', '#E2364A')
ggtree(tree,layout="roundrect")%<+% phylopic_info +
    geom_nodepoint() +
    geom_tiplab(aes(image=phylopic),color=color, geom="phylopic", offset=0)+
    geom_tiplab(aes(label=common_name), offset = 2.2) + layout_dendrogram()


plotBreakLongEdges(tree,n=3); axisPhylo();nodelabels(frame ="circle",bg ="black",font=0,cex =.2)