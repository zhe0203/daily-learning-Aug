library(shiny)


setwd("C:/Users/asus/Desktop")
df = read.csv('data1.csv')

df$name = as.character(df$name)

a = df$name[!duplicated(df$name)]


ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      radioButtons("show_vars", "Distribution type:",
                   c("customer" = "customer",
                     "ent_info" = "ent_info",
                     "employee" = "employee",
                     "employee2" = "employee2",
                     "employee3" = "employee3",
                     "ent_info1" = "ent_info1",
                     "ent_info2" = "ent_info2"))
    ),
    mainPanel(
      tableOutput("view")
    )
  )
)


server <- function(input, output) {
  datasetInput <- reactive({
    switch(input$show_vars,
           "customer" = df[df$name == 'customer',],
           "ent_info" = df[df$name == 'ent_info',],
           "employee" = df[df$name == 'employee',],
           "employee2" = df[df$name == 'employee2',],
           "employee3" = df[df$name == 'employee3',],
           "ent_info1" = df[df$name == 'ent_info1',],
           "ent_info2" = df[df$name == 'ent_info2',])
  })

  output$view <- renderTable({
    datasetInput()
  })
  
}

shinyApp(ui, server)
