# Load R packages
suppressMessages(suppressWarnings(library(glue)))
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(ggtext)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(shadowtext)))
suppressMessages(suppressWarnings(library(colorspace)))
suppressMessages(suppressWarnings(library(data.table)))

# Set Env
no <- basename(dirname(rstudioapi::getActiveDocumentContext()$path))
wkdir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(wkdir)

# Load file
d <- fread('data/most_visited_nps_species_data.csv')

# Arrange data
d$CommonNames <- NULL
d$Synonyms    <- NULL

d1 <- d[, by = .(ParkName, CategoryName), .(
  N = SciName |> unique() |> length()
)]

mm = d1 |> dcast(CategoryName ~ ParkName, value.var = "N", fill = 0)
mm = mm[, -1] |> setDF(rownames = mm$CategoryName) |> as.matrix()

o_r <- mm |> dist("euclidean") |> hclust("ward.D2")
o_c <- mm |> t() |> dist("euclidean") |> hclust("ward.D2")

d1$ParkName <- d1$ParkName |> factor(levels = o_c$labels[o_c$order])
d1$CategoryName <- d1$CategoryName |> factor(levels = o_r$labels[o_r$order])

# Plotting constants
base.size   <- 7
text.size   <- 5 / ggplot2::.pt
base.family <- 'sans'

# Draw plots
gg <- d1 |>
  ggplot(aes(ParkName, CategoryName)) + 
  
  geom_vline(xintercept = seq(0.5, 15.5, by = 1), linewidth = .25, color = "grey") +
  geom_hline(yintercept = seq(0.5, 16.5, by = 1), linewidth = .25, color = "grey") +
  
  geom_tile(aes(fill = N), linewidth = .25, color = "grey20") +
  
  geom_shadowtext(
    aes(color = N, label = N |> scales::comma()), fontface = "bold",
    bg.color = "grey96", bg.r = .1, size = text.size, family = base.family
  ) +
  
  scale_fill_stepsn(
    colors = c("#00429d","#73a2c6","#ffffe0","#f4777f","#93003a") |> lighten(.1),
    breaks = c(10, 100, 1000, 10000),
    transform = "log10",
    labels = scales::comma,
    guide = guide_colorsteps(
      barheight = unit(16, "lines"), 
      barwidth = unit(.5, "lines")
    )
  ) +
  
  scale_color_stepsn(
    colors = c("#00429d","#73a2c6","#ffffe0","#f4777f","#93003a") |> darken(.75),
    breaks = c(10, 100, 1000, 10000),
    transform = "log10",
    guide = "none"
  ) +
  
  # facet_wrap(vars(Nativeness)) +
  
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  
  theme_minimal(base_size = base.size, base_family = base.family) +
  
  theme(
    legend.title.position = "left",
    legend.title = element_text(angle = 90, hjust = .5, face = "bold", size = base.size, family = base.family),
    
    axis.title.y = element_text(face = "bold", margin = margin(r = 10)),
    axis.title.x = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    
    panel.grid = element_blank(),
    panel.background = element_rect(fill = "grey95", color = NA),
    
    plot.title    = element_text(face = "bold", size = base.size, margin = margin(b = 5), hjust = .5, family = base.family),
    plot.subtitle = element_markdown(margin = margin(b = 8), size = base.size, color = "grey25", hjust = .5, family = base.family),
    
    plot.background = element_rect(fill = "#FFFFFF", color = NA),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  
  labs(
    fill = "No. of species", x = "National Park Full Name", y = "Species Category",
    
    title = "National Park Species",
    
    subtitle = paste0(
      "**Species distribution** by category across U.S. National Parks."
    )
  )

# Save to file
width = 140; height = 140
ggsave(gg, filename = glue('{wkdir}/{no}.png'), width = width, height = height, units = 'mm', dpi = 300, device = 'png', bg = '#FFFFFF')
ggsave(gg, filename = glue('{wkdir}/{no}.pdf'), width = width, height = height, units = 'mm', dpi = 300, device = cairo_pdf, bg = '#FFFFFF')

# sessionInfo
sink(glue('{wkdir}/{no}.txt'))
sessionInfo()
sink()