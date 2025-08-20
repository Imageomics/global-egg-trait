# imageomics 

## import
metadata_inat_with_geo <- read.csv("~/Documents/Imageomics/metadata_inat_with_geo.csv")
View(metadata_inat_with_geo)

# install.packages("elevatr")
library(elevatr)
# library(sf)
# library(sp)
df <- data.frame(metadata_inat_with_geo$longitude, metadata_inat_with_geo$latitude)
df.nona <- na.omit(df)
colnames(df.nona) <- c("x","y")
ll_prj <- 4326
metadata_with_geo_elevation <- get_elev_point(df.nona, prj = ll_prj, src = "aws")

# then I just added it to the other metadata file and added NA's back in
